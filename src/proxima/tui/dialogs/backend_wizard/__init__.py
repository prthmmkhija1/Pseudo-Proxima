"""Backend Addition Wizard - Phases 1-8 Implementation.

This package provides a step-by-step wizard for adding custom quantum
simulator backends to Proxima. The wizard guides users through:

1. Welcome & Backend Type Selection
2. Basic Information Input
3. Capabilities Configuration
4. Gate Mapping Configuration
5. Code Template Generation
6. Testing & Validation
7. Change Management & Approval
8. Advanced Testing & Validation
9. Final Deployment & Success Confirmation
10. Review & Deployment

Phase 2 adds the backend infrastructure:
- Change tracking with undo/redo
- Code preview dialogs
- Deployment success/failure dialogs
- Change review screens

Phase 4 adds Testing & Validation Interface:
- Comprehensive test execution
- Circuit selection and shots configuration
- Real-time progress display
- Detailed results reporting

Phase 5 adds Integration & Deployment:
- Enhanced review step with full summary
- Deployment progress tracking
- File writing with backup/rollback
- Registry integration

Phase 6 adds Change Management & Approval:
- Dedicated step for reviewing AI-generated changes
- Approval workflow with policies and audit trail
- Individual change approval/rejection
- Diff preview for each change
- Export changes to JSON/patch format
- Change statistics and progress tracking

Phase 7 adds Advanced Testing & Validation:
- Comprehensive test suite with 4 categories
- Unit tests, integration tests, performance tests, compatibility tests
- Real-time progress visualization with pause/resume/skip
- Detailed test logging and report generation
- Export reports to JSON and HTML
- Performance metrics collection

Phase 8 adds Final Deployment & Success Confirmation:
- Deployment complete screen with success celebration
- Deployment summary with files created, tests passed
- Backend details display
- Next steps guidance
- Quick actions for testing, documentation, export
- Quick test screen for post-deployment validation
- Documentation viewer for generated docs
- Deployment verifier for automated checks

Usage:
    from proxima.tui.dialogs.backend_wizard import BackendWizardManager
    
    wizard = BackendWizardManager()
    await wizard.start()
"""

from .wizard_state import BackendWizardState
from .wizard_manager import BackendWizardManager
from .step_welcome import WelcomeStepScreen
from .step_basic_info import BasicInfoStepScreen
from .step_capabilities import CapabilitiesStepScreen
from .step_gate_mapping import GateMappingStepScreen
from .step_code_template import CodeTemplateStepScreen
from .step_testing import TestingStepScreen
from .step_review import ReviewStepScreen

# Phase 2: Backend Configuration Interface
from .change_tracker import ChangeTracker, FileChange, ChangeType
from .code_preview_dialog import CodePreviewDialog, SingleFilePreviewDialog
from .deployment_success_dialog import DeploymentSuccessDialog, DeploymentFailureDialog
from .change_review_screen import ChangeReviewScreen

# Phase 4: Testing & Validation Interface (enhanced step)
from .step_testing_enhanced import (
    TestingStepScreen as EnhancedTestingStepScreen,
    TestCategoryDisplay,
    TestResultsWidget,
)

# Phase 5: Integration & Deployment
from .step_review_enhanced import (
    EnhancedReviewStepScreen,
    DeploymentProgressWidget,
    FilesListWidget,
    BackendSummaryWidget,
)
from .deployment_progress_dialog import (
    DeploymentProgressDialog,
    DeploymentStageWidget,
    QuickDeployDialog,
)

# Phase 6: Change Management & Approval System
from .step_change_management import (
    ChangeManagementStepScreen,
    ChangeItemWidget,
    DiffPreviewWidget,
    ChangeStatsWidget,
)
from .approval_workflow import (
    ApprovalWorkflowManager,
    ApprovalRequest,
    ApprovalPolicy,
    ApprovalStatus,
    ApprovalCategory,
    ChangeApprovalIntegrator,
)

# Phase 7: Advanced Testing & Validation
from .advanced_testing import (
    AdvancedTestingScreen,
    AdvancedTestResult,
    TestCategory,
    TestCategoryWidget,
    OverallProgressWidget,
    TestLogWidget,
    ComprehensiveTestRunner,
)
from .test_report_viewer import (
    TestReportViewer,
    TestReportSummaryWidget,
    CategoryBreakdownWidget,
    FailedTestsWidget,
    PerformanceMetricsWidget,
    TestResultDetailsDialog,
)
from .comprehensive_test_runner import (
    TestRunner,
    UnitTestRunner,
    IntegrationTestRunner,
    PerformanceTestRunner,
    CompatibilityTestRunner,
    FullTestSuite,
    TestLevel,
    TestDefinition,
    TestExecutionResult,
)

# Phase 8: Final Deployment & Success Confirmation
from .deployment_complete_screen import (
    DeploymentCompleteScreen,
    DeploymentSummaryWidget,
    BackendDetailsWidget,
    NextStepsWidget,
    QuickActionsWidget,
)
from .quick_test_screen import (
    QuickTestScreen,
    QuickTestProgressWidget,
    QuickTestResultsWidget,
    CircuitOutputWidget,
    QuickTestResult,
)
from .documentation_viewer import (
    DocumentationViewer,
    CodeDocumentationViewer,
)
from .deployment_verifier import (
    DeploymentVerifier,
    QuickVerifier,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
    create_verification_summary,
)

__all__ = [
    # Phase 1: Wizard Steps
    "BackendWizardState",
    "BackendWizardManager",
    "WelcomeStepScreen",
    "BasicInfoStepScreen",
    "CapabilitiesStepScreen",
    "GateMappingStepScreen",
    "CodeTemplateStepScreen",
    "TestingStepScreen",
    "ReviewStepScreen",
    # Phase 2: Backend Configuration Interface
    "ChangeTracker",
    "FileChange",
    "ChangeType",
    "CodePreviewDialog",
    "SingleFilePreviewDialog",
    "DeploymentSuccessDialog",
    "DeploymentFailureDialog",
    "ChangeReviewScreen",
    # Phase 4: Testing & Validation Interface
    "EnhancedTestingStepScreen",
    "TestCategoryDisplay",
    "TestResultsWidget",
    # Phase 5: Integration & Deployment
    "EnhancedReviewStepScreen",
    "DeploymentProgressWidget",
    "FilesListWidget",
    "BackendSummaryWidget",
    "DeploymentProgressDialog",
    "DeploymentStageWidget",
    "QuickDeployDialog",
    # Phase 6: Change Management & Approval System
    "ChangeManagementStepScreen",
    "ChangeItemWidget",
    "DiffPreviewWidget",
    "ChangeStatsWidget",
    "ApprovalWorkflowManager",
    "ApprovalRequest",
    "ApprovalPolicy",
    "ApprovalStatus",
    "ApprovalCategory",
    "ChangeApprovalIntegrator",
    # Phase 7: Advanced Testing & Validation
    "AdvancedTestingScreen",
    "AdvancedTestResult",
    "TestCategory",
    "TestCategoryWidget",
    "OverallProgressWidget",
    "TestLogWidget",
    "ComprehensiveTestRunner",
    "TestReportViewer",
    "TestReportSummaryWidget",
    "CategoryBreakdownWidget",
    "FailedTestsWidget",
    "PerformanceMetricsWidget",
    "TestResultDetailsDialog",
    "TestRunner",
    "UnitTestRunner",
    "IntegrationTestRunner",
    "PerformanceTestRunner",
    "CompatibilityTestRunner",
    "FullTestSuite",
    "TestLevel",
    "TestDefinition",
    "TestExecutionResult",
    # Phase 8: Final Deployment & Success Confirmation
    "DeploymentCompleteScreen",
    "DeploymentSummaryWidget",
    "BackendDetailsWidget",
    "NextStepsWidget",
    "QuickActionsWidget",
    "QuickTestScreen",
    "QuickTestProgressWidget",
    "QuickTestResultsWidget",
    "CircuitOutputWidget",
    "QuickTestResult",
    "DocumentationViewer",
    "CodeDocumentationViewer",
    "DeploymentVerifier",
    "QuickVerifier",
    "VerificationReport",
    "VerificationResult",
    "VerificationStatus",
    "create_verification_summary",
]
