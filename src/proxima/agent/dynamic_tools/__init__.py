"""Dynamic Tool System for Proxima AI Assistant.

This module provides a fully dynamic, AI-reasoning-powered tool system that replaces
hardcoded keyword matching with LLM-driven tool selection and execution.

Phase 1: Foundation - Dynamic Tool System Architecture
======================================================
- Tool Registry and Discovery System
- Dynamic Tool Execution Engine  
- Tool Implementation Library

The system enables any integrated LLM (Ollama, Gemini, GPT, Claude, etc.) to:
1. Understand user intent through natural language
2. Dynamically select appropriate tools based on reasoning
3. Extract and validate parameters intelligently
4. Execute operations with proper context and error handling
5. Generate context-aware responses

Architecture Components:
-----------------------
1. Tool Interface (tool_interface.py)
   - Defines ToolInterface abstract base class
   - ToolDefinition for LLM-readable tool schemas
   - Multi-provider support (OpenAI, Anthropic, Gemini)

2. Tool Registry (tool_registry.py)  
   - Central registry for dynamic tool discovery
   - Semantic search for tool selection
   - @register_tool decorator for auto-registration

3. Execution Context (execution_context.py)
   - Carries state across tool executions
   - Git state, file access, conversation history
   - Snapshot support for rollback capability

4. Tool Orchestrator (tool_orchestrator.py)
   - Multi-tool execution coordination
   - Dependency resolution and parallel execution
   - Error handling and recovery

5. Progress Tracker (progress_tracker.py)
   - Real-time progress events
   - Callbacks for UI integration

6. Result Processor (result_processor.py)
   - Standardizes tool output formats
   - LLM-optimized context generation

7. LLM Integration (llm_integration.py)
   - Provider-agnostic bridge to LLMs
   - Parses tool calls from various formats
   - Formats results for consumption

8. Tools (tools/)
   - FileSystem, Git, Terminal tools
   - Self-registering via @register_tool
"""

from .tool_interface import (
    ToolInterface,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolCategory,
    PermissionLevel,
    RiskLevel,
    ParameterType,
    BaseTool,
)

from .tool_registry import (
    ToolRegistry,
    RegisteredTool,
    ToolSearchResult,
    get_tool_registry,
    register_tool,
)

from .execution_context import (
    ExecutionContext,
    ContextSnapshot,
    ContextManager,
    GitState,
    TerminalSession,
    FileSystemState,
    ConversationMessage,
    UserPreferences,
    SystemInfo,
    get_context_manager,
    get_current_context,
)

from .tool_orchestrator import (
    ToolOrchestrator,
    ExecutionPlan,
    ExecutionStep,
    ExecutionStepStatus,
    ExecutionPlanStatus,
    OrchestratorConfig,
    get_tool_orchestrator,
)

from .progress_tracker import (
    ProgressTracker,
    TrackedOperation,
    ProgressEvent,
    ProgressEventType,
    ProgressStatus,
    get_progress_tracker,
)

from .result_processor import (
    ResultProcessor,
    ProcessedResult,
    AggregatedResult,
    ResultType,
    ResultSeverity,
    get_result_processor,
)

from .llm_integration import (
    LLMToolIntegration,
    LLMToolConfig,
    LLMProvider,
    ToolCall,
    ToolCallResult,
    get_llm_tool_integration,
    configure_llm_integration,
)

# Phase 2: LLM Integration - Natural Language to Tool Mapping
from .intent_classifier import (
    IntentClassifier,
    ClassifiedIntent,
    IntentType,
    IntentConfidence,
    ClassifierConfig,
    get_intent_classifier,
)

from .entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    EntityType,
    ExtractionResult,
    ExtractorConfig,
    get_entity_extractor,
)

from .structured_output import (
    StructuredOutputParser,
    JSONExtractor,
    SchemaValidator,
    ParseResult,
    OutputFormat,
)

from .function_calling import (
    FunctionCallingIntegration,
    FunctionDefinitionGenerator,
    FunctionCallParser,
    FunctionCallExecutor,
    FunctionCallFormat,
    FunctionCall,
    FunctionCallResult,
    get_function_calling_integration,
)

from .provider_router import (
    ProviderRouter,
    ProviderAdapter,
    ProviderType,
    ProviderCapability,
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    RouterConfig,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    AnthropicAdapter,
    get_provider_router,
    configure_provider_router,
)

from .natural_language_processor import (
    NaturalLanguageProcessor,
    SyncNaturalLanguageProcessor,
    ProcessingContext,
    ProcessingResult,
    ProcessingStage,
    get_nl_processor,
    process_query,
)

# Phase 3: Advanced Reasoning - Multi-Step and Complex Operations
from .task_decomposer import (
    TaskDecomposer,
    TaskNode,
    DependencyGraph,
    DecompositionResult,
    DependencyEdge,
    TaskType,
    TaskPriority,
    ResourceType,
    get_task_decomposer,
)

from .workflow_engine import (
    WorkflowEngine,
    Workflow,
    WorkflowNode,
    WorkflowTemplate,
    Transition,
    WorkflowVariable,
    WorkflowStatus,
    NodeType,
    TransitionType,
    get_workflow_engine,
)

from .adaptive_executor import (
    AdaptiveExecutor,
    TaskExecution,
    ExecutionProgress,
    ExecutionMetrics,
    ExecutionConfig,
    ResourceInfo,
    ExecutionStrategy,
    RetryStrategy,
    FallbackStrategy,
    ResourceState,
    get_adaptive_executor,
)

from .build_system import (
    BuildSystem,
    RepositoryAnalysis,
    BuildEnvironment,
    BuildResult,
    BuildFile,
    Dependency,
    TechnologyStack,
    BuildSystemType,
    BuildStatus,
    get_build_system,
)

from .terminal_manager import (
    TerminalManager,
    TerminalSession as ManagedTerminalSession,
    ProcessInfo,
    OutputChunk,
    TerminalConfig,
    TerminalState,
    ProcessState,
    OutputType,
    get_terminal_manager,
)

from .result_analyzer import (
    ResultAnalyzer,
    FormatDetectionResult,
    ExtractedValue,
    ExtractionResult,
    StatisticalResult,
    TrendResult,
    AnomalyResult,
    AnalysisResult,
    Report,
    OutputFormat as AnalysisOutputFormat,
    DataType,
    AnalysisType,
    ReportFormat,
    get_result_analyzer,
)

# Phase 4: GitHub Integration and Authentication
from .github_auth import (
    GitHubAuthenticator,
    OAuthConfig,
    TokenInfo,
    SSHKeyInfo,
    AuthSession,
    AuditLogEntry,
    CredentialStore,
    CredentialEncryptor,
    AuthMethod,
    AuthStatus,
    TokenScope,
    get_github_authenticator,
)

from .github_repo_ops import (
    GitHubRepoOperations,
    GitHubAPIClient,
    RepositoryInfo,
    IssueInfo,
    ReleaseInfo,
    SearchResult,
    IssueState,
    PRMergeMethod,
    RepoVisibility,
    SortDirection,
    RepoSearchSort,
    get_github_repo_operations,
)

from .github_actions import (
    GitHubActionsManager,
    GitHubActionsAPI,
    WorkflowInfo,
    WorkflowRunInfo,
    JobInfo,
    ArtifactInfo,
    WorkflowDispatchInput,
    WorkflowStatus,
    WorkflowConclusion,
    JobStatus,
    ArtifactExpiration,
    get_github_actions_manager,
)

# Phase 5: File System Intelligence
from .file_intelligence import (
    FileSystemIntelligence,
    FuzzyPathResolver,
    ContentAwareFileHandler,
    SafeFileOperations,
    SmartFileSearch,
    PathMatch,
    FileInfo,
    SearchResult,
    OperationJournalEntry,
    FileIndex,
    FileCategory,
    ChangeType,
    ConflictStrategy,
    SearchType,
    get_file_system_intelligence,
)

from .file_monitor import (
    FileMonitorManager,
    FileSystemWatcher,
    ChangeDetector,
    FileSynchronizer,
    FileEvent,
    FileChange,
    SyncConflict,
    SyncState,
    EventType,
    ChangeCategory,
    SyncDirection,
    MergeStrategy,
    get_file_monitor_manager,
)

# Phase 6: Advanced Git Operations
from .git_workflows import (
    GitWorkflowManager,
    SmartBranchManager,
    CommitIntelligence,
    MergeConflictResolver,
    HistoryManager,
    GitCommandRunner,
    BranchInfo,
    CommitInfo,
    ConflictInfo,
    MergeResult,
    BisectResult,
    BranchType,
    ConflictStrategy as GitConflictStrategy,
    CommitType,
    MergeReadiness,
    BisectState,
    get_git_workflow_manager,
)

from .git_remote_manager import (
    GitRemoteManager,
    RemoteConfigManager,
    IntelligentPushPull,
    FetchOptimizer,
    RemoteInfo,
    PushResult,
    PullResult,
    FetchResult,
    ConflictPrediction,
    RemoteType,
    ProtocolType,
    PushProtection,
    FetchStrategy,
    SyncStatus,
    get_git_remote_manager,
)

# Phase 7: Dynamic Configuration and Preferences
from .config_preferences import (
    ActionTracker,
    PreferenceLearner,
    ContextAwareDefaults,
    PersonalizationEngine,
    AdaptiveConfigurationManager,
    UserAction,
    LearnedPreference,
    ContextDefaults,
    WorkflowPattern,
    PreferenceCategory,
    PreferenceStrength,
    OperationMode,
    EnvironmentType as PreferenceEnvironmentType,
    get_adaptive_configuration_manager,
)

from .config_manager import (
    ProfileManager,
    EnvironmentDetector,
    ConfigurationSynchronizer,
    ConfigurationManager,
    ConfigValue,
    ConfigProfile,
    ConfigVersion,
    ConfigConflict,
    AuditEntry,
    ProfileType,
    ConfigScope,
    EnvironmentType as ConfigEnvironmentType,
    SyncStatus as ConfigSyncStatus,
    ConflictResolution,
    get_configuration_manager,
)

# Phase 8: Error Handling and Recovery
from .error_detection import (
    IntelligentErrorDetector,
    ErrorClassifier,
    ErrorContextCapture,
    ErrorAnalyzer,
    ErrorExplainer,
    ErrorContext,
    ErrorPattern,
    AnalyzedError,
    ErrorCluster,
    ErrorSeverity,
    ErrorCategory,
    ErrorState,
    get_intelligent_error_detector,
)

from .error_recovery import (
    AutomatedRecoverySystem,
    CircuitBreaker,
    RetryExecutor,
    CheckpointManager,
    FallbackManager,
    RecoveryStrategySelector,
    RetryConfig,
    RecoveryAttempt,
    Checkpoint,
    StrategyRecord,
    RecoveryResult,
    RecoveryStrategy,
    RetryBackoff,
    CircuitState,
    RecoveryState,
    get_automated_recovery_system,
    with_retry,
)

from .error_prevention import (
    ProactiveIssuePrevention,
    PreflightValidator,
    HealthMonitor,
    WarningSystem,
    ValidationCheck,
    PreflightResult,
    Warning,
    ResourceMetrics,
    HealthReport,
    ValidationResult,
    RiskLevel,
    WarningSeverity,
    HealthStatus,
    ResourceType as PreventionResourceType,
    get_proactive_issue_prevention,
)

# Phase 9: Testing and Validation Framework
from .testing_framework import (
    ComprehensiveTestingFramework,
    TestRunner,
    TestGenerator,
    IntegrationTestRunner,
    LLMResponseTester,
    PerformanceTester,
    CoverageTracker,
    MockObject,
    TestCase,
    TestSuite,
    TestFixture,
    MockCall,
    CoverageReport,
    PerformanceMetrics,
    LLMTestResult,
    TestStatus,
    TestPriority,
    TestCategory,
    MockBehavior,
    get_comprehensive_testing_framework,
    test_case,
    fixture,
)

from .quality_assurance import (
    QualityAssuranceAutomation,
    StaticAnalyzer,
    SecurityScanner,
    TypeChecker,
    CIPipelineRunner,
    ValidationSuiteRunner,
    AnalysisIssue,
    AnalysisReport,
    CIJob,
    CIPipeline,
    ValidationScenario,
    AnalysisSeverity,
    AnalysisCategory,
    ValidationStatus,
    PlatformType,
    get_quality_assurance_automation,
)

# Phase 10: Deployment and Monitoring
from .deployment_monitoring import (
    DeploymentAndMonitoring,
    DeploymentPipeline,
    ConfigurationManager as DeploymentConfigManager,
    DependencyManager,
    StructuredLogger,
    MetricsCollector,
    DistributedTracer,
    HealthChecker,
    AlertManager,
    UsageAnalytics,
    FeedbackManager,
    ModelPerformanceMonitor,
    ExperimentFramework,
    DeploymentConfig,
    DeploymentResult,
    EnvironmentConfig,
    DependencyInfo,
    LogEntry,
    Metric,
    TraceSpan,
    HealthCheckResult,
    Alert,
    UsageEvent,
    UserFeedback,
    ModelPerformanceMetric,
    DeploymentStrategy,
    DeploymentStatus,
    HealthStatus as DeploymentHealthStatus,
    AlertSeverity,
    MetricType,
    LogLevel,
    FeedbackType,
    get_deployment_and_monitoring,
)

# Phase 11: Security and Compliance
from .security_compliance import (
    SecurityAndCompliance,
    AuthenticationManager,
    AuthorizationManager,
    DataProtectionManager,
    InputValidator,
    SecurityScanner as ComplianceSecurityScanner,
    AuditLogger,
    PrivacyManager,
    ComplianceManager,
    User,
    Session,
    RoleDefinition,
    EncryptedData,
    ThreatDetection,
    Vulnerability,
    AuditEvent,
    Consent,
    DataSubjectRequest,
    ComplianceCheck,
    AuthenticationMethod,
    AuthenticationStatus,
    Role,
    Permission,
    EncryptionAlgorithm,
    ThreatType,
    VulnerabilitySeverity,
    ComplianceStandard,
    AuditEventType,
    ConsentType,
    DataSubjectRequestType,
    get_security_and_compliance,
)

# Phase 12: Documentation and Knowledge Management
from .documentation_knowledge import (
    DocumentationAndKnowledge,
    UserDocumentationManager,
    DeveloperDocumentationManager,
    SystemDocumentationManager,
    DocumentationIndexer,
    ContextualHelpSystem,
    KnowledgeBaseManager,
    DocumentationEntry,
    Tutorial,
    FAQEntry,
    TroubleshootingGuide,
    APIDocumentation,
    KnowledgeArticle,
    SearchResult as DocSearchResult,
    ContextualHelp,
    HelpHistoryEntry,
    Bookmark,
    DocumentationMetrics,
    DocumentationType,
    DocumentationFormat,
    HelpLevel,
    SearchRelevance,
    DocumentationStatus,
    KnowledgeCategory,
    TutorialDifficulty,
    get_documentation_and_knowledge,
)

# Robust NL Processor for dynamic intent recognition
from .robust_nl_processor import (
    RobustNLProcessor,
    IntentType,
    Intent,
    ExtractedEntity,
    SessionContext,
    get_robust_nl_processor,
)

# Import tools to trigger auto-registration
from . import tools

__all__ = [
    # Core interfaces
    "ToolInterface",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "BaseTool",
    
    # Enums
    "ToolCategory",
    "PermissionLevel",
    "RiskLevel",
    "ParameterType",
    "ExecutionStepStatus",
    "ExecutionPlanStatus",
    "ProgressStatus",
    "ProgressEventType",
    "ResultType",
    "ResultSeverity",
    "LLMProvider",
    
    # Registry
    "ToolRegistry",
    "RegisteredTool",
    "ToolSearchResult",
    "get_tool_registry",
    "register_tool",
    
    # Context
    "ExecutionContext",
    "ContextSnapshot",
    "ContextManager",
    "GitState",
    "TerminalSession",
    "FileSystemState",
    "ConversationMessage",
    "UserPreferences",
    "SystemInfo",
    "get_context_manager",
    "get_current_context",
    
    # Orchestration
    "ToolOrchestrator",
    "ExecutionPlan",
    "ExecutionStep",
    "OrchestratorConfig",
    "get_tool_orchestrator",
    
    # Progress
    "ProgressTracker",
    "TrackedOperation",
    "ProgressEvent",
    "get_progress_tracker",
    
    # Result processing
    "ResultProcessor",
    "ProcessedResult",
    "AggregatedResult",
    "get_result_processor",
    
    # LLM Integration
    "LLMToolIntegration",
    "LLMToolConfig",
    "ToolCall",
    "ToolCallResult",
    "get_llm_tool_integration",
    "configure_llm_integration",
    
    # Phase 2: Intent Classification
    "IntentClassifier",
    "ClassifiedIntent",
    "IntentType",
    "IntentConfidence",
    "ClassifierConfig",
    "get_intent_classifier",
    
    # Phase 2: Entity Extraction
    "EntityExtractor",
    "ExtractedEntity",
    "EntityType",
    "ExtractionResult",
    "ExtractorConfig",
    "get_entity_extractor",
    
    # Phase 2: Structured Output
    "StructuredOutputParser",
    "JSONExtractor",
    "SchemaValidator",
    "ParseResult",
    "OutputFormat",
    
    # Phase 2: Function Calling
    "FunctionCallingIntegration",
    "FunctionDefinitionGenerator",
    "FunctionCallParser",
    "FunctionCallExecutor",
    "FunctionCallFormat",
    "FunctionCall",
    "FunctionCallResult",
    "get_function_calling_integration",
    
    # Phase 2: Provider Router
    "ProviderRouter",
    "ProviderAdapter",
    "ProviderType",
    "ProviderCapability",
    "ProviderConfig",
    "ProviderRequest",
    "ProviderResponse",
    "RouterConfig",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "get_provider_router",
    "configure_provider_router",
    
    # Phase 2: Natural Language Processor
    "NaturalLanguageProcessor",
    "SyncNaturalLanguageProcessor",
    "ProcessingContext",
    "ProcessingResult",
    "ProcessingStage",
    "get_nl_processor",
    "process_query",
    
    # Phase 3: Task Decomposition
    "TaskDecomposer",
    "TaskNode",
    "DependencyGraph",
    "DecompositionResult",
    "DependencyEdge",
    "TaskType",
    "TaskPriority",
    "ResourceType",
    "get_task_decomposer",
    
    # Phase 3: Workflow Engine
    "WorkflowEngine",
    "Workflow",
    "WorkflowNode",
    "WorkflowTemplate",
    "Transition",
    "WorkflowVariable",
    "WorkflowStatus",
    "NodeType",
    "TransitionType",
    "get_workflow_engine",
    
    # Phase 3: Adaptive Executor
    "AdaptiveExecutor",
    "TaskExecution",
    "ExecutionProgress",
    "ExecutionMetrics",
    "ExecutionConfig",
    "ResourceInfo",
    "ExecutionStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    "ResourceState",
    "get_adaptive_executor",
    
    # Phase 3: Build System
    "BuildSystem",
    "RepositoryAnalysis",
    "BuildEnvironment",
    "BuildResult",
    "BuildFile",
    "Dependency",
    "TechnologyStack",
    "BuildSystemType",
    "BuildStatus",
    "get_build_system",
    
    # Phase 3: Terminal Manager
    "TerminalManager",
    "ManagedTerminalSession",
    "ProcessInfo",
    "OutputChunk",
    "TerminalConfig",
    "TerminalState",
    "ProcessState",
    "OutputType",
    "get_terminal_manager",
    
    # Phase 3: Result Analyzer
    "ResultAnalyzer",
    "FormatDetectionResult",
    "ExtractedValue",
    "ExtractionResult",
    "StatisticalResult",
    "TrendResult",
    "AnomalyResult",
    "AnalysisResult",
    "Report",
    "AnalysisOutputFormat",
    "DataType",
    "AnalysisType",
    "ReportFormat",
    "get_result_analyzer",
    
    # Phase 4: GitHub Authentication
    "GitHubAuthenticator",
    "OAuthConfig",
    "TokenInfo",
    "SSHKeyInfo",
    "AuthSession",
    "AuditLogEntry",
    "CredentialStore",
    "CredentialEncryptor",
    "AuthMethod",
    "AuthStatus",
    "TokenScope",
    "get_github_authenticator",
    
    # Phase 4: GitHub Repository Operations
    "GitHubRepoOperations",
    "GitHubAPIClient",
    "RepositoryInfo",
    "IssueInfo",
    "ReleaseInfo",
    "SearchResult",
    "IssueState",
    "PRMergeMethod",
    "RepoVisibility",
    "SortDirection",
    "RepoSearchSort",
    "get_github_repo_operations",
    
    # Phase 4: GitHub Actions
    "GitHubActionsManager",
    "GitHubActionsAPI",
    "WorkflowInfo",
    "WorkflowRunInfo",
    "JobInfo",
    "ArtifactInfo",
    "WorkflowDispatchInput",
    "WorkflowStatus",
    "WorkflowConclusion",
    "JobStatus",
    "ArtifactExpiration",
    "get_github_actions_manager",
    
    # Phase 5: File System Intelligence
    "FileSystemIntelligence",
    "FuzzyPathResolver",
    "ContentAwareFileHandler",
    "SafeFileOperations",
    "SmartFileSearch",
    "PathMatch",
    "FileInfo",
    "SearchResult",
    "OperationJournalEntry",
    "FileIndex",
    "FileCategory",
    "ChangeType",
    "ConflictStrategy",
    "SearchType",
    "get_file_system_intelligence",
    
    # Phase 5: File Monitoring
    "FileMonitorManager",
    "FileSystemWatcher",
    "ChangeDetector",
    "FileSynchronizer",
    "FileEvent",
    "FileChange",
    "SyncConflict",
    "SyncState",
    "EventType",
    "ChangeCategory",
    "SyncDirection",
    "MergeStrategy",
    "get_file_monitor_manager",
    
    # Phase 6: Git Workflows
    "GitWorkflowManager",
    "SmartBranchManager",
    "CommitIntelligence",
    "MergeConflictResolver",
    "HistoryManager",
    "GitCommandRunner",
    "BranchInfo",
    "CommitInfo",
    "ConflictInfo",
    "MergeResult",
    "BisectResult",
    "BranchType",
    "GitConflictStrategy",
    "CommitType",
    "MergeReadiness",
    "BisectState",
    "get_git_workflow_manager",
    
    # Phase 6: Git Remote Management
    "GitRemoteManager",
    "RemoteConfigManager",
    "IntelligentPushPull",
    "FetchOptimizer",
    "RemoteInfo",
    "PushResult",
    "PullResult",
    "FetchResult",
    "ConflictPrediction",
    "RemoteType",
    "ProtocolType",
    "PushProtection",
    "FetchStrategy",
    "SyncStatus",
    "get_git_remote_manager",
    
    # Phase 7: Adaptive Configuration (config_preferences.py)
    "ActionTracker",
    "PreferenceLearner",
    "ContextAwareDefaults",
    "PersonalizationEngine",
    "AdaptiveConfigurationManager",
    "UserAction",
    "LearnedPreference",
    "ContextDefaults",
    "WorkflowPattern",
    "PreferenceCategory",
    "PreferenceStrength",
    "OperationMode",
    "PreferenceEnvironmentType",
    "get_adaptive_configuration_manager",
    
    # Phase 7: Configuration Management (config_manager.py)
    "ProfileManager",
    "EnvironmentDetector",
    "ConfigurationSynchronizer",
    "ConfigurationManager",
    "ConfigValue",
    "ConfigProfile",
    "ConfigVersion",
    "ConfigConflict",
    "AuditEntry",
    "ProfileType",
    "ConfigScope",
    "ConfigEnvironmentType",
    "ConfigSyncStatus",
    "ConflictResolution",
    "get_configuration_manager",
    
    # Phase 8: Intelligent Error Detection (error_detection.py)
    "IntelligentErrorDetector",
    "ErrorClassifier",
    "ErrorContextCapture",
    "ErrorAnalyzer",
    "ErrorExplainer",
    "ErrorContext",
    "ErrorPattern",
    "AnalyzedError",
    "ErrorCluster",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorState",
    "get_intelligent_error_detector",
    
    # Phase 8: Automated Recovery (error_recovery.py)
    "AutomatedRecoverySystem",
    "CircuitBreaker",
    "RetryExecutor",
    "CheckpointManager",
    "FallbackManager",
    "RecoveryStrategySelector",
    "RetryConfig",
    "RecoveryAttempt",
    "Checkpoint",
    "StrategyRecord",
    "RecoveryResult",
    "RecoveryStrategy",
    "RetryBackoff",
    "CircuitState",
    "RecoveryState",
    "get_automated_recovery_system",
    "with_retry",
    
    # Phase 8: Proactive Issue Prevention (error_prevention.py)
    "ProactiveIssuePrevention",
    "PreflightValidator",
    "HealthMonitor",
    "WarningSystem",
    "ValidationCheck",
    "PreflightResult",
    "Warning",
    "ResourceMetrics",
    "HealthReport",
    "ValidationResult",
    "RiskLevel",
    "WarningSeverity",
    "HealthStatus",
    "PreventionResourceType",
    "get_proactive_issue_prevention",
    
    # Phase 9: Testing Framework (testing_framework.py)
    "ComprehensiveTestingFramework",
    "TestRunner",
    "TestGenerator",
    "IntegrationTestRunner",
    "LLMResponseTester",
    "PerformanceTester",
    "CoverageTracker",
    "MockObject",
    "TestCase",
    "TestSuite",
    "TestFixture",
    "MockCall",
    "CoverageReport",
    "PerformanceMetrics",
    "LLMTestResult",
    "TestStatus",
    "TestPriority",
    "TestCategory",
    "MockBehavior",
    "get_comprehensive_testing_framework",
    "test_case",
    "fixture",
    
    # Phase 9: Quality Assurance (quality_assurance.py)
    "QualityAssuranceAutomation",
    "StaticAnalyzer",
    "SecurityScanner",
    "TypeChecker",
    "CIPipelineRunner",
    "ValidationSuiteRunner",
    "AnalysisIssue",
    "AnalysisReport",
    "CIJob",
    "CIPipeline",
    "ValidationScenario",
    "AnalysisSeverity",
    "AnalysisCategory",
    "ValidationStatus",
    "PlatformType",
    "get_quality_assurance_automation",
    
    # Phase 10: Deployment and Monitoring (deployment_monitoring.py)
    "DeploymentAndMonitoring",
    "DeploymentPipeline",
    "DeploymentConfigManager",
    "DependencyManager",
    "StructuredLogger",
    "MetricsCollector",
    "DistributedTracer",
    "HealthChecker",
    "AlertManager",
    "UsageAnalytics",
    "FeedbackManager",
    "ModelPerformanceMonitor",
    "ExperimentFramework",
    "DeploymentConfig",
    "DeploymentResult",
    "EnvironmentConfig",
    "DependencyInfo",
    "LogEntry",
    "Metric",
    "TraceSpan",
    "HealthCheckResult",
    "Alert",
    "UsageEvent",
    "UserFeedback",
    "ModelPerformanceMetric",
    "DeploymentStrategy",
    "DeploymentStatus",
    "DeploymentHealthStatus",
    "AlertSeverity",
    "MetricType",
    "LogLevel",
    "FeedbackType",
    "get_deployment_and_monitoring",
    
    # Phase 11: Security and Compliance (security_compliance.py)
    "SecurityAndCompliance",
    "AuthenticationManager",
    "AuthorizationManager",
    "DataProtectionManager",
    "InputValidator",
    "ComplianceSecurityScanner",
    "AuditLogger",
    "PrivacyManager",
    "ComplianceManager",
    "User",
    "Session",
    "RoleDefinition",
    "EncryptedData",
    "ThreatDetection",
    "Vulnerability",
    "AuditEvent",
    "Consent",
    "DataSubjectRequest",
    "ComplianceCheck",
    "AuthenticationMethod",
    "AuthenticationStatus",
    "Role",
    "Permission",
    "EncryptionAlgorithm",
    "ThreatType",
    "VulnerabilitySeverity",
    "ComplianceStandard",
    "AuditEventType",
    "ConsentType",
    "DataSubjectRequestType",
    "get_security_and_compliance",
    
    # Phase 12: Documentation and Knowledge Management (documentation_knowledge.py)
    "DocumentationAndKnowledge",
    "UserDocumentationManager",
    "DeveloperDocumentationManager",
    "SystemDocumentationManager",
    "DocumentationIndexer",
    "ContextualHelpSystem",
    "KnowledgeBaseManager",
    "DocumentationEntry",
    "Tutorial",
    "FAQEntry",
    "TroubleshootingGuide",
    "APIDocumentation",
    "KnowledgeArticle",
    "DocSearchResult",
    "ContextualHelp",
    "HelpHistoryEntry",
    "Bookmark",
    "DocumentationMetrics",
    "DocumentationType",
    "DocumentationFormat",
    "HelpLevel",
    "SearchRelevance",
    "DocumentationStatus",
    "KnowledgeCategory",
    "TutorialDifficulty",
    "get_documentation_and_knowledge",
    
    # Robust NL Processor
    "RobustNLProcessor",
    "IntentType",
    "Intent",
    "ExtractedEntity",
    "SessionContext",
    "get_robust_nl_processor",
]


def initialize_dynamic_tools():
    """Initialize the dynamic tools system.
    
    Call this once at application startup to set up the global
    instances and ensure all tools are registered.
    
    Returns:
        dict: Dictionary with all initialized components:
            - registry: Tool registry
            - context_manager: Execution context manager
            - orchestrator: Tool orchestrator
            - nl_processor: Natural language processor (Phase 2)
            - provider_router: LLM provider router (Phase 2)
            - task_decomposer: Task decomposition engine (Phase 3)
            - workflow_engine: Workflow engine (Phase 3)
            - adaptive_executor: Adaptive executor (Phase 3)
            - build_system: Build system (Phase 3)
            - terminal_manager: Terminal manager (Phase 3)
            - result_analyzer: Result analyzer (Phase 3)
            - github_auth: GitHub authenticator (Phase 4)
            - github_repo_ops: GitHub repository operations (Phase 4)
            - github_actions: GitHub Actions manager (Phase 4)
            - file_intelligence: File system intelligence (Phase 5)
            - file_monitor: File monitoring manager (Phase 5)
            - git_workflow: Git workflow manager (Phase 6)
            - git_remote: Git remote manager (Phase 6)
            - adaptive_config: Adaptive configuration manager (Phase 7)
            - config_manager: Configuration manager (Phase 7)
            - error_detector: Intelligent error detector (Phase 8)
            - recovery_system: Automated recovery system (Phase 8)
            - issue_prevention: Proactive issue prevention (Phase 8)
            - testing_framework: Comprehensive testing framework (Phase 9)
            - qa_automation: Quality assurance automation (Phase 9)
            - deployment_monitoring: Deployment and monitoring system (Phase 10)
            - security_compliance: Security and compliance system (Phase 11)
            - documentation_knowledge: Documentation and knowledge management (Phase 12)
    """
    registry = get_tool_registry()
    context_manager = get_context_manager()
    orchestrator = get_tool_orchestrator()
    
    # Phase 2: Initialize NL processing components
    nl_processor = get_nl_processor(tool_registry=registry)
    provider_router = get_provider_router()
    
    # Phase 3: Initialize advanced reasoning components
    task_decomposer = get_task_decomposer()
    workflow_engine = get_workflow_engine()
    adaptive_executor = get_adaptive_executor()
    build_system = get_build_system()
    terminal_manager = get_terminal_manager()
    result_analyzer = get_result_analyzer()
    
    # Phase 4: Initialize GitHub integration components
    github_auth = get_github_authenticator()
    github_repo_ops = get_github_repo_operations(authenticator=github_auth)
    github_actions = get_github_actions_manager(authenticator=github_auth)
    
    # Phase 5: Initialize file system intelligence components
    file_intelligence = get_file_system_intelligence()
    file_monitor = get_file_monitor_manager()
    
    # Phase 6: Initialize advanced git operations components
    git_workflow = get_git_workflow_manager()
    git_remote = get_git_remote_manager()
    
    # Phase 7: Initialize configuration and preferences components
    adaptive_config = get_adaptive_configuration_manager()
    config_manager = get_configuration_manager()
    
    # Phase 8: Initialize error handling and recovery components
    error_detector = get_intelligent_error_detector()
    recovery_system = get_automated_recovery_system()
    issue_prevention = get_proactive_issue_prevention()
    
    # Phase 9: Initialize testing and quality assurance components
    testing_framework = get_comprehensive_testing_framework()
    qa_automation = get_quality_assurance_automation()
    
    # Phase 10: Initialize deployment and monitoring components
    deployment_monitoring = get_deployment_and_monitoring()
    
    # Phase 11: Initialize security and compliance components
    security_compliance = get_security_and_compliance()
    
    # Phase 12: Initialize documentation and knowledge management components
    documentation_knowledge = get_documentation_and_knowledge()
    
    # Tools are auto-registered via @register_tool decorator
    # when the tools module is imported above
    
    return {
        "registry": registry,
        "context_manager": context_manager,
        "orchestrator": orchestrator,
        "nl_processor": nl_processor,
        "provider_router": provider_router,
        # Phase 3 components
        "task_decomposer": task_decomposer,
        "workflow_engine": workflow_engine,
        "adaptive_executor": adaptive_executor,
        "build_system": build_system,
        "terminal_manager": terminal_manager,
        "result_analyzer": result_analyzer,
        # Phase 4 components
        "github_auth": github_auth,
        "github_repo_ops": github_repo_ops,
        "github_actions": github_actions,
        # Phase 5 components
        "file_intelligence": file_intelligence,
        "file_monitor": file_monitor,
        # Phase 6 components
        "git_workflow": git_workflow,
        "git_remote": git_remote,
        # Phase 7 components
        "adaptive_config": adaptive_config,
        "config_manager": config_manager,
        # Phase 8 components
        "error_detector": error_detector,
        "recovery_system": recovery_system,
        "issue_prevention": issue_prevention,
        # Phase 9 components
        "testing_framework": testing_framework,
        "qa_automation": qa_automation,
        # Phase 10 components
        "deployment_monitoring": deployment_monitoring,
        # Phase 11 components
        "security_compliance": security_compliance,
        # Phase 12 components
        "documentation_knowledge": documentation_knowledge,
    }


def get_tool_count() -> int:
    """Get the number of registered tools."""
    return len(get_tool_registry())


def list_available_tools() -> list:
    """List all available tool names."""
    return [t.definition.name for t in get_tool_registry()]


def process_natural_language(query: str, sync: bool = True):
    """Process a natural language query using Phase 2 components.
    
    This is a convenience function for quick access to NL processing.
    
    Args:
        query: The natural language query
        sync: If True, use synchronous processing
        
    Returns:
        ProcessingResult with the response and tool results
    """
    if sync:
        processor = SyncNaturalLanguageProcessor(
            tool_registry=get_tool_registry()
        )
        return processor.process(query)
    else:
        import asyncio
        return asyncio.run(process_query(query, tool_registry=get_tool_registry()))
