"""Backend Wizard Manager.

Orchestrates the multi-step wizard flow for adding custom backends.
Manages navigation between steps and coordinates state updates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable, TYPE_CHECKING
from enum import IntEnum

from textual.app import App
from textual.screen import Screen

from .wizard_state import BackendWizardState

if TYPE_CHECKING:
    from textual.app import App


class WizardStep(IntEnum):
    """Wizard step enumeration."""
    WELCOME = 1
    BASIC_INFO = 2
    CAPABILITIES = 3
    GATE_MAPPING = 4
    CODE_TEMPLATE = 5
    TESTING = 6
    REVIEW = 7


class BackendWizardManager:
    """
    Manages the backend addition wizard workflow.
    
    Coordinates the multi-step wizard, handling navigation between
    steps, state management, and final backend generation.
    
    Usage:
        manager = BackendWizardManager(app)
        await manager.start()
    """
    
    def __init__(
        self,
        app: App,
        on_complete: Optional[Callable[[BackendWizardState], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the wizard manager.
        
        Args:
            app: The Textual application instance
            on_complete: Callback when wizard completes successfully
            on_cancel: Callback when wizard is cancelled
        """
        self.app = app
        self.state = BackendWizardState()
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self._step_screens: Dict[int, Screen] = {}
    
    async def start(self) -> None:
        """Start the wizard from the first step."""
        self.state.reset()
        await self._show_step(WizardStep.WELCOME)
    
    async def _show_step(self, step: WizardStep) -> None:
        """
        Show a specific wizard step.
        
        Args:
            step: The step to display
        """
        self.state.current_step = step.value
        
        # Import step screens dynamically to avoid circular imports
        screen = self._create_step_screen(step)
        
        if screen:
            result = await self.app.push_screen_wait(screen)
            await self._handle_step_result(step, result)
    
    def _create_step_screen(self, step: WizardStep) -> Optional[Screen]:
        """
        Create the screen for a specific step.
        
        Args:
            step: The wizard step
            
        Returns:
            The screen instance for the step
        """
        if step == WizardStep.WELCOME:
            from .step_welcome import WelcomeStepScreen
            return WelcomeStepScreen(self.state)
        
        elif step == WizardStep.BASIC_INFO:
            from .step_basic_info import BasicInfoStepScreen
            return BasicInfoStepScreen(self.state)
        
        elif step == WizardStep.CAPABILITIES:
            from .step_capabilities import CapabilitiesStepScreen
            return CapabilitiesStepScreen(self.state)
        
        elif step == WizardStep.GATE_MAPPING:
            from .step_gate_mapping import GateMappingStepScreen
            return GateMappingStepScreen(self.state)
        
        elif step == WizardStep.CODE_TEMPLATE:
            from .step_code_template import CodeTemplateStepScreen
            return CodeTemplateStepScreen(self.state)
        
        elif step == WizardStep.TESTING:
            from .step_testing import TestingStepScreen
            return TestingStepScreen(self.state)
        
        elif step == WizardStep.REVIEW:
            from .step_review import ReviewStepScreen
            return ReviewStepScreen(self.state)
        
        return None
    
    async def _handle_step_result(
        self,
        step: WizardStep,
        result: Optional[Dict[str, Any]]
    ) -> None:
        """
        Handle the result from a wizard step.
        
        Args:
            step: The step that completed
            result: The result dictionary from the step
        """
        if result is None:
            # Screen was dismissed without result
            return
        
        action = result.get("action", "")
        
        if action == "cancel":
            await self._handle_cancel()
        
        elif action == "back":
            if step.value > 1:
                prev_step = WizardStep(step.value - 1)
                await self._show_step(prev_step)
        
        elif action == "next":
            # Update state from result
            if "state" in result:
                self.state = result["state"]
            
            # Move to next step or complete
            if step == WizardStep.REVIEW:
                await self._complete_wizard()
            elif step.value < 7:
                next_step = WizardStep(step.value + 1)
                await self._show_step(next_step)
        
        elif action == "complete":
            await self._complete_wizard()
    
    async def _handle_cancel(self) -> None:
        """Handle wizard cancellation."""
        # Confirm cancellation
        from ..confirmation import ConfirmationDialog
        
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Cancel Backend Wizard",
                message="Are you sure you want to cancel?\nAll progress will be lost.",
                confirm_label="Yes, Cancel",
                cancel_label="Continue Wizard"
            )
        )
        
        if confirmed:
            self.state.reset()
            if self.on_cancel:
                self.on_cancel()
    
    async def _complete_wizard(self) -> None:
        """Complete the wizard and generate the backend."""
        try:
            # Generate backend code
            from .backend_generator import BackendCodeGenerator
            
            generator = BackendCodeGenerator(self.state)
            success, paths, contents = generator.generate_all_files()
            
            if success:
                self.state.generated_code = contents
                self.state.output_directory = str(generator.output_dir)
                
                # Notify completion
                self.app.notify(
                    f"Backend '{self.state.display_name}' created successfully!",
                    severity="information"
                )
                
                if self.on_complete:
                    self.on_complete(self.state)
            else:
                self.app.notify(
                    "Failed to generate backend code",
                    severity="error"
                )
        
        except Exception as e:
            self.app.notify(
                f"Error generating backend: {str(e)}",
                severity="error"
            )
    
    def get_progress(self) -> tuple[int, int, float]:
        """
        Get wizard progress information.
        
        Returns:
            Tuple of (current_step, total_steps, percentage)
        """
        current = self.state.current_step
        total = 7
        percentage = (current / total) * 100
        return current, total, percentage
    
    def get_step_title(self, step: Optional[int] = None) -> str:
        """
        Get the title for a wizard step.
        
        Args:
            step: Step number (1-7), or current step if None
            
        Returns:
            Human-readable step title
        """
        step = step or self.state.current_step
        titles = {
            1: "Welcome",
            2: "Basic Information",
            3: "Capabilities",
            4: "Gate Mapping",
            5: "Code Template",
            6: "Testing",
            7: "Review & Deploy",
        }
        return titles.get(step, "Unknown")
