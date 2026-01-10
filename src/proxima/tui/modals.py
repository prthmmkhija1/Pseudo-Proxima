"""TUI Modals and Dialogs for Proxima.

Step 6.1: Modal dialogs including:
- Confirmation dialogs
- Input dialogs
- Progress dialogs
- Error dialogs
- Consent dialogs
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ProgressBar, Switch


class DialogResult(Enum):
    """Result of a dialog interaction."""

    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    DISMISSED = "dismissed"


@dataclass
class ModalResponse:
    """Response from a modal dialog."""

    result: DialogResult
    data: dict[str, Any] | None = None

    @property
    def confirmed(self) -> bool:
        """Check if dialog was confirmed."""
        return self.result == DialogResult.CONFIRMED

    @property
    def cancelled(self) -> bool:
        """Check if dialog was cancelled."""
        return self.result == DialogResult.CANCELLED


# ========== Base Modal ==========


class BaseModal(ModalScreen[ModalResponse]):
    """Base class for modal dialogs."""

    DEFAULT_CSS = """
    BaseModal {
        align: center middle;
    }

    BaseModal > Container {
        width: auto;
        max-width: 80%;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 2;
    }

    BaseModal .modal-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
        color: $primary;
    }

    BaseModal .modal-content {
        margin: 1 0;
    }

    BaseModal .modal-buttons {
        margin-top: 2;
        align: center middle;
        height: auto;
    }

    BaseModal Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    def __init__(
        self,
        title: str = "Dialog",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title

    def action_cancel(self) -> None:
        """Cancel the modal."""
        self.dismiss(ModalResponse(result=DialogResult.CANCELLED))

    def action_confirm(self) -> None:
        """Confirm the modal."""
        self.dismiss(ModalResponse(result=DialogResult.CONFIRMED))


# ========== Confirmation Modal ==========


class ConfirmModal(BaseModal):
    """Confirmation dialog with yes/no buttons."""

    DEFAULT_CSS = """
    ConfirmModal > Container {
        min-width: 40;
    }

    ConfirmModal .confirm-message {
        text-align: center;
        margin: 1 2;
    }
    """

    def __init__(
        self,
        title: str = "Confirm",
        message: str = "Are you sure?",
        confirm_label: str = "Yes",
        cancel_label: str = "No",
        confirm_variant: str = "primary",
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._message = message
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self._title, classes="modal-title")
            yield Label(self._message, classes="confirm-message")
            with Horizontal(classes="modal-buttons"):
                yield Button(
                    self._cancel_label,
                    id="btn-cancel",
                    variant="default",
                )
                yield Button(
                    self._confirm_label,
                    id="btn-confirm",
                    variant=self._confirm_variant,
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()


# ========== Input Modal ==========


class InputModal(BaseModal):
    """Dialog for text input."""

    DEFAULT_CSS = """
    InputModal > Container {
        min-width: 50;
    }

    InputModal .input-label {
        margin-bottom: 1;
    }

    InputModal Input {
        margin: 1 0;
    }
    """

    def __init__(
        self,
        title: str = "Input",
        label: str = "Enter value:",
        placeholder: str = "",
        default_value: str = "",
        password: bool = False,
        validator: Callable[[str], bool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._label = label
        self._placeholder = placeholder
        self._default_value = default_value
        self._password = password
        self._validator = validator
        self._input_value = default_value

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self._title, classes="modal-title")
            yield Label(self._label, classes="input-label")
            yield Input(
                value=self._default_value,
                placeholder=self._placeholder,
                password=self._password,
                id="modal-input",
            )
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="btn-cancel")
                yield Button("OK", id="btn-confirm", variant="primary")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Track input value changes."""
        self._input_value = event.value

    def action_confirm(self) -> None:
        """Confirm with validation."""
        if self._validator and not self._validator(self._input_value):
            self.notify("Invalid input", severity="error")
            return

        self.dismiss(
            ModalResponse(
                result=DialogResult.CONFIRMED,
                data={"value": self._input_value},
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()


# ========== Choice Modal ==========


class ChoiceModal(BaseModal):
    """Dialog for selecting from multiple options."""

    DEFAULT_CSS = """
    ChoiceModal > Container {
        min-width: 50;
    }

    ChoiceModal .choice-list {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }

    ChoiceModal .choice-item {
        padding: 1;
        margin: 0 0 1 0;
        border: solid $surface-light;
    }

    ChoiceModal .choice-item:hover {
        border: solid $primary;
        background: $surface-light;
    }

    ChoiceModal .choice-item.selected {
        border: solid $primary;
        background: $primary-dark;
    }
    """

    def __init__(
        self,
        title: str = "Select",
        message: str = "Choose an option:",
        choices: list[str] | None = None,
        default_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._message = message
        self._choices = choices or []
        self._selected_index = default_index

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self._title, classes="modal-title")
            yield Label(self._message)
            with Vertical(classes="choice-list"):
                for i, choice in enumerate(self._choices):
                    classes = "choice-item"
                    if i == self._selected_index:
                        classes += " selected"
                    yield Button(choice, id=f"choice-{i}", classes=classes)
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="btn-cancel")
                yield Button("Select", id="btn-confirm", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id and event.button.id.startswith("choice-"):
            # Select the choice
            index = int(event.button.id.split("-")[1])
            self._select_choice(index)
        elif event.button.id == "btn-confirm":
            self.dismiss(
                ModalResponse(
                    result=DialogResult.CONFIRMED,
                    data={
                        "index": self._selected_index,
                        "choice": self._choices[self._selected_index] if self._choices else None,
                    },
                )
            )
        else:
            self.action_cancel()

    def _select_choice(self, index: int) -> None:
        """Update selected choice."""
        # Remove selected class from current
        for btn in self.query(".choice-item"):
            btn.remove_class("selected")

        # Add to new selection
        new_btn = self.query_one(f"#choice-{index}", Button)
        new_btn.add_class("selected")
        self._selected_index = index


# ========== Progress Modal ==========


class ProgressModal(BaseModal):
    """Dialog showing progress of an operation."""

    DEFAULT_CSS = """
    ProgressModal > Container {
        min-width: 60;
    }

    ProgressModal .progress-message {
        text-align: center;
        margin-bottom: 1;
    }

    ProgressModal .progress-status {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    ProgressModal ProgressBar {
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    class ProgressUpdated(Message):
        """Message when progress is updated."""

        def __init__(self, progress: float, status: str = "") -> None:
            super().__init__()
            self.progress = progress
            self.status = status

    def __init__(
        self,
        title: str = "Progress",
        message: str = "Processing...",
        cancellable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._message = message
        self._cancellable = cancellable
        self._progress = 0.0
        self._status = ""

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self._title, classes="modal-title")
            yield Label(self._message, classes="progress-message", id="progress-msg")
            yield ProgressBar(total=100, id="progress-bar")
            yield Label("", classes="progress-status", id="progress-status")
            if self._cancellable:
                with Horizontal(classes="modal-buttons"):
                    yield Button("Cancel", id="btn-cancel", variant="error")

    def update_progress(self, progress: float, status: str = "") -> None:
        """Update progress value and status.

        Args:
            progress: Progress percentage (0-100)
            status: Optional status message
        """
        self._progress = progress
        self._status = status

        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.update(progress=progress)

        if status:
            status_label = self.query_one("#progress-status", Label)
            status_label.update(status)

    def complete(self, message: str = "Complete!") -> None:
        """Mark progress as complete."""
        self.update_progress(100, message)
        self.dismiss(ModalResponse(result=DialogResult.CONFIRMED))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle cancel button."""
        if event.button.id == "btn-cancel":
            self.action_cancel()


# ========== Error Modal ==========


class ErrorModal(BaseModal):
    """Dialog for displaying errors."""

    DEFAULT_CSS = """
    ErrorModal > Container {
        min-width: 50;
        border: thick $error;
    }

    ErrorModal .modal-title {
        color: $error;
    }

    ErrorModal .error-message {
        margin: 1 0;
        padding: 1;
        background: $background;
        border: solid $error;
    }

    ErrorModal .error-details {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close", show=True),
        Binding("enter", "dismiss_modal", "Close", show=False),
    ]

    def __init__(
        self,
        title: str = "Error",
        message: str = "An error occurred",
        details: str = "",
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._message = message
        self._details = details

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(f"❌ {self._title}", classes="modal-title")
            yield Label(self._message, classes="error-message")
            if self._details:
                yield Label(self._details, classes="error-details")
            with Horizontal(classes="modal-buttons"):
                yield Button("Close", id="btn-close", variant="error")

    def action_dismiss_modal(self) -> None:
        """Dismiss the error modal."""
        self.dismiss(ModalResponse(result=DialogResult.DISMISSED))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle close button."""
        self.action_dismiss_modal()


# ========== Consent Modal ==========


class ConsentModal(BaseModal):
    """Dialog for requesting user consent."""

    DEFAULT_CSS = """
    ConsentModal > Container {
        min-width: 60;
        max-width: 80;
    }

    ConsentModal .consent-header {
        text-style: bold;
        margin-bottom: 1;
    }

    ConsentModal .consent-description {
        margin: 1 0;
        padding: 1;
        background: $background;
        border: solid $warning;
    }

    ConsentModal .consent-implications {
        margin: 1 0;
    }

    ConsentModal .consent-implication {
        padding-left: 2;
        color: $text-muted;
    }

    ConsentModal .consent-checkbox {
        margin: 1 0;
        padding: 1;
        border: solid $surface-light;
    }

    ConsentModal .force-warning {
        color: $error;
        text-style: bold;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        title: str = "Consent Required",
        operation: str = "perform this action",
        description: str = "",
        implications: list[str] | None = None,
        allow_force: bool = True,
        require_acknowledgment: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._operation = operation
        self._description = description
        self._implications = implications or []
        self._allow_force = allow_force
        self._require_acknowledgment = require_acknowledgment
        self._acknowledged = not require_acknowledgment
        self._force = False

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(f"⚠️ {self._title}", classes="modal-title")
            yield Label(
                f"You are about to: {self._operation}",
                classes="consent-header",
            )

            if self._description:
                yield Label(self._description, classes="consent-description")

            if self._implications:
                yield Label("This action will:", classes="consent-implications")
                for imp in self._implications:
                    yield Label(f"• {imp}", classes="consent-implication")

            if self._require_acknowledgment:
                with Horizontal(classes="consent-checkbox"):
                    yield Switch(id="acknowledge-switch")
                    yield Label("I understand the implications")

            if self._allow_force:
                with Horizontal(classes="consent-checkbox"):
                    yield Switch(id="force-switch")
                    yield Label("Force execution (skip safety checks)")
                yield Label(
                    "⚠️ Force mode bypasses safety checks!",
                    classes="force-warning",
                )

            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="btn-cancel")
                yield Button(
                    "Proceed",
                    id="btn-confirm",
                    variant="warning",
                    disabled=self._require_acknowledgment,
                )

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "acknowledge-switch":
            self._acknowledged = event.value
            confirm_btn = self.query_one("#btn-confirm", Button)
            confirm_btn.disabled = not self._acknowledged
        elif event.switch.id == "force-switch":
            self._force = event.value

    def action_confirm(self) -> None:
        """Confirm consent."""
        self.dismiss(
            ModalResponse(
                result=DialogResult.CONFIRMED,
                data={
                    "acknowledged": self._acknowledged,
                    "force": self._force,
                },
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()


# ========== Multi-Input Modal ==========


@dataclass
class FormField:
    """Definition of a form field."""

    key: str
    label: str
    field_type: str = "text"  # text, password, switch, select
    default: Any = ""
    placeholder: str = ""
    options: list[str] | None = None  # For select fields
    required: bool = False


class FormModal(BaseModal):
    """Dialog with multiple input fields."""

    DEFAULT_CSS = """
    FormModal > Container {
        min-width: 60;
    }

    FormModal .form-field {
        margin: 1 0;
    }

    FormModal .form-label {
        margin-bottom: 0;
    }

    FormModal .form-required {
        color: $error;
    }

    FormModal Input {
        margin-top: 1;
    }

    FormModal .form-error {
        color: $error;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        title: str = "Form",
        fields: list[FormField] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title=title, **kwargs)
        self._fields = fields or []
        self._values: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

        # Initialize default values
        for field in self._fields:
            self._values[field.key] = field.default

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self._title, classes="modal-title")

            for field in self._fields:
                with Vertical(classes="form-field"):
                    required_mark = " *" if field.required else ""
                    yield Label(
                        f"{field.label}{required_mark}",
                        classes="form-label",
                    )

                    if field.field_type == "text":
                        yield Input(
                            value=str(field.default),
                            placeholder=field.placeholder,
                            id=f"field-{field.key}",
                        )
                    elif field.field_type == "password":
                        yield Input(
                            value=str(field.default),
                            placeholder=field.placeholder,
                            password=True,
                            id=f"field-{field.key}",
                        )
                    elif field.field_type == "switch":
                        yield Switch(
                            value=bool(field.default),
                            id=f"field-{field.key}",
                        )

            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="btn-cancel")
                yield Button("Submit", id="btn-confirm", variant="primary")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Track input changes."""
        if event.input.id and event.input.id.startswith("field-"):
            key = event.input.id[6:]  # Remove "field-" prefix
            self._values[key] = event.value

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Track switch changes."""
        if event.switch.id and event.switch.id.startswith("field-"):
            key = event.switch.id[6:]
            self._values[key] = event.value

    def _validate(self) -> bool:
        """Validate all fields."""
        self._errors.clear()
        valid = True

        for field in self._fields:
            value = self._values.get(field.key, "")

            if field.required:
                if not value and value != False:  # noqa: E712
                    self._errors[field.key] = f"{field.label} is required"
                    valid = False

        if not valid:
            error_msg = "; ".join(self._errors.values())
            self.notify(error_msg, severity="error")

        return valid

    def action_confirm(self) -> None:
        """Confirm with validation."""
        if not self._validate():
            return

        self.dismiss(
            ModalResponse(
                result=DialogResult.CONFIRMED,
                data=self._values.copy(),
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()


# ========== Modal Factory ==========


async def show_confirm(
    app,
    title: str = "Confirm",
    message: str = "Are you sure?",
    confirm_label: str = "Yes",
    cancel_label: str = "No",
) -> bool:
    """Show a confirmation dialog.

    Args:
        app: The Textual App instance
        title: Dialog title
        message: Confirmation message
        confirm_label: Label for confirm button
        cancel_label: Label for cancel button

    Returns:
        True if confirmed, False otherwise
    """
    modal = ConfirmModal(
        title=title,
        message=message,
        confirm_label=confirm_label,
        cancel_label=cancel_label,
    )
    response = await app.push_screen_wait(modal)
    return response.confirmed if response else False


async def show_input(
    app,
    title: str = "Input",
    label: str = "Enter value:",
    placeholder: str = "",
    default_value: str = "",
    password: bool = False,
) -> str | None:
    """Show an input dialog.

    Args:
        app: The Textual App instance
        title: Dialog title
        label: Input field label
        placeholder: Placeholder text
        default_value: Default input value
        password: Whether to mask input

    Returns:
        Input value if confirmed, None otherwise
    """
    modal = InputModal(
        title=title,
        label=label,
        placeholder=placeholder,
        default_value=default_value,
        password=password,
    )
    response = await app.push_screen_wait(modal)
    if response and response.confirmed and response.data:
        return response.data.get("value")
    return None


async def show_error(
    app,
    title: str = "Error",
    message: str = "An error occurred",
    details: str = "",
) -> None:
    """Show an error dialog.

    Args:
        app: The Textual App instance
        title: Dialog title
        message: Error message
        details: Additional error details
    """
    modal = ErrorModal(title=title, message=message, details=details)
    await app.push_screen_wait(modal)


async def show_consent(
    app,
    operation: str,
    description: str = "",
    implications: list[str] | None = None,
    allow_force: bool = True,
) -> ModalResponse | None:
    """Show a consent dialog.

    Args:
        app: The Textual App instance
        operation: Description of the operation
        description: Detailed description
        implications: List of implications
        allow_force: Whether to allow force option

    Returns:
        ModalResponse with consent data, or None if cancelled
    """
    modal = ConsentModal(
        operation=operation,
        description=description,
        implications=implications,
        allow_force=allow_force,
    )
    return await app.push_screen_wait(modal)
