"""CLI Interactive Prompts - User prompts, confirmations, and consent dialogs.

This module provides:
- Prompt: Base class for user prompts
- ConfirmPrompt: Yes/no confirmation
- SelectPrompt: Single selection from options
- MultiSelectPrompt: Multiple selection from options
- TextPrompt: Free text input
- PasswordPrompt: Hidden password input
- ConsentPrompt: Consent request with explanation
- prompt_*: Convenience functions for common prompts
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from getpass import getpass
from typing import Any, Generic, TypeVar

import typer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt as RichPrompt
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ========== Prompt Result ==========


class PromptResult(Enum):
    """Result of a prompt operation."""

    ANSWERED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()
    DEFAULT = auto()


@dataclass
class PromptResponse(Generic[TypeVar("T")]):
    """Response from a prompt."""

    value: Any
    result: PromptResult = PromptResult.ANSWERED
    used_default: bool = False

    @property
    def cancelled(self) -> bool:
        return self.result == PromptResult.CANCELLED

    @property
    def answered(self) -> bool:
        return self.result == PromptResult.ANSWERED


# ========== Base Prompt ==========


T = TypeVar("T")


class Prompt(ABC, Generic[T]):
    """Base class for user prompts."""

    def __init__(
        self,
        message: str,
        default: T | None = None,
        required: bool = False,
        validator: Callable[[T], bool] | None = None,
        error_message: str = "Invalid input",
    ) -> None:
        self.message = message
        self.default = default
        self.required = required
        self.validator = validator
        self.error_message = error_message
        self._console = Console() if RICH_AVAILABLE else None

    @abstractmethod
    def ask(self) -> PromptResponse[T]:
        """Ask the user for input."""
        pass

    def _validate(self, value: T) -> bool:
        """Validate the input value."""
        if self.validator:
            return self.validator(value)
        return True


# ========== Confirm Prompt ==========


class ConfirmPrompt(Prompt[bool]):
    """Yes/no confirmation prompt."""

    def __init__(
        self,
        message: str,
        default: bool = False,
        yes_text: str = "y",
        no_text: str = "n",
    ) -> None:
        super().__init__(message, default=default)
        self.yes_text = yes_text
        self.no_text = no_text

    def ask(self) -> PromptResponse[bool]:
        """Ask for confirmation."""
        try:
            if RICH_AVAILABLE and self._console:
                result = Confirm.ask(
                    self.message,
                    default=self.default,
                    console=self._console,
                )
            else:
                default_str = f"[{self.yes_text}/{self.no_text}]"
                if self.default is not None:
                    default_indicator = (
                        self.yes_text.upper() if self.default else self.no_text.upper()
                    )
                    default_str = f"[{default_indicator}]"

                response = input(f"{self.message} {default_str}: ").strip().lower()

                if not response and self.default is not None:
                    return PromptResponse(self.default, PromptResult.DEFAULT, used_default=True)

                result = response in ("y", "yes", "true", "1")

            return PromptResponse(result, PromptResult.ANSWERED)

        except (KeyboardInterrupt, EOFError):
            return PromptResponse(False, PromptResult.CANCELLED)


# ========== Text Prompt ==========


class TextPrompt(Prompt[str]):
    """Free text input prompt."""

    def __init__(
        self,
        message: str,
        default: str | None = None,
        placeholder: str = "",
        min_length: int = 0,
        max_length: int | None = None,
    ) -> None:
        super().__init__(message, default=default)
        self.placeholder = placeholder
        self.min_length = min_length
        self.max_length = max_length

    def ask(self) -> PromptResponse[str]:
        """Ask for text input."""
        try:
            if RICH_AVAILABLE and self._console:
                result = RichPrompt.ask(
                    self.message,
                    default=self.default or "",
                    console=self._console,
                )
            else:
                default_str = f" [{self.default}]" if self.default else ""
                result = input(f"{self.message}{default_str}: ").strip()

                if not result and self.default:
                    return PromptResponse(self.default, PromptResult.DEFAULT, used_default=True)

            # Validate length
            if len(result) < self.min_length:
                typer.echo(f"Input too short (min {self.min_length} characters)")
                return self.ask()

            if self.max_length and len(result) > self.max_length:
                typer.echo(f"Input too long (max {self.max_length} characters)")
                return self.ask()

            return PromptResponse(result, PromptResult.ANSWERED)

        except (KeyboardInterrupt, EOFError):
            return PromptResponse("", PromptResult.CANCELLED)


# ========== Password Prompt ==========


class PasswordPrompt(Prompt[str]):
    """Hidden password input prompt."""

    def __init__(
        self,
        message: str = "Password",
        confirm: bool = False,
        confirm_message: str = "Confirm password",
    ) -> None:
        super().__init__(message)
        self.confirm = confirm
        self.confirm_message = confirm_message

    def ask(self) -> PromptResponse[str]:
        """Ask for password input."""
        try:
            if RICH_AVAILABLE and self._console:
                password = RichPrompt.ask(
                    self.message,
                    password=True,
                    console=self._console,
                )

                if self.confirm:
                    confirm_password = RichPrompt.ask(
                        self.confirm_message,
                        password=True,
                        console=self._console,
                    )
                    if password != confirm_password:
                        self._console.print("[red]Passwords do not match[/red]")
                        return self.ask()
            else:
                password = getpass(f"{self.message}: ")

                if self.confirm:
                    confirm_password = getpass(f"{self.confirm_message}: ")
                    if password != confirm_password:
                        print("Passwords do not match")
                        return self.ask()

            return PromptResponse(password, PromptResult.ANSWERED)

        except (KeyboardInterrupt, EOFError):
            return PromptResponse("", PromptResult.CANCELLED)


# ========== Select Prompt ==========


@dataclass
class SelectOption:
    """An option for selection prompts."""

    value: str
    label: str = ""
    description: str = ""
    disabled: bool = False

    def __post_init__(self):
        if not self.label:
            self.label = self.value


class SelectPrompt(Prompt[str]):
    """Single selection from options prompt."""

    def __init__(
        self,
        message: str,
        options: list[str] | list[SelectOption],
        default: str | None = None,
    ) -> None:
        super().__init__(message, default=default)

        # Convert string options to SelectOption
        self.options: list[SelectOption] = []
        for opt in options:
            if isinstance(opt, str):
                self.options.append(SelectOption(value=opt, label=opt))
            else:
                self.options.append(opt)

    def ask(self) -> PromptResponse[str]:
        """Ask for selection."""
        try:
            # Display options
            if RICH_AVAILABLE and self._console:
                self._console.print(f"\n[bold]{self.message}[/bold]")
                for i, opt in enumerate(self.options, 1):
                    disabled_marker = " [dim](disabled)[/dim]" if opt.disabled else ""
                    self._console.print(f"  {i}. {opt.label}{disabled_marker}")
                    if opt.description:
                        self._console.print(f"     [dim]{opt.description}[/dim]")
            else:
                print(f"\n{self.message}")
                for i, opt in enumerate(self.options, 1):
                    disabled_marker = " (disabled)" if opt.disabled else ""
                    print(f"  {i}. {opt.label}{disabled_marker}")

            # Get selection
            default_str = ""
            if self.default:
                for i, opt in enumerate(self.options, 1):
                    if opt.value == self.default:
                        default_str = f" [{i}]"
                        break

            response = input(f"Select (1-{len(self.options)}){default_str}: ").strip()

            if not response and self.default:
                return PromptResponse(self.default, PromptResult.DEFAULT, used_default=True)

            try:
                index = int(response) - 1
                if 0 <= index < len(self.options):
                    opt = self.options[index]
                    if opt.disabled:
                        print("That option is disabled")
                        return self.ask()
                    return PromptResponse(opt.value, PromptResult.ANSWERED)
            except ValueError:
                # Try matching by value
                for opt in self.options:
                    if opt.value.lower() == response.lower():
                        if opt.disabled:
                            print("That option is disabled")
                            return self.ask()
                        return PromptResponse(opt.value, PromptResult.ANSWERED)

            print("Invalid selection")
            return self.ask()

        except (KeyboardInterrupt, EOFError):
            return PromptResponse("", PromptResult.CANCELLED)


# ========== Multi-Select Prompt ==========


class MultiSelectPrompt(Prompt[list[str]]):
    """Multiple selection from options prompt."""

    def __init__(
        self,
        message: str,
        options: list[str] | list[SelectOption],
        default: list[str] | None = None,
        min_selections: int = 0,
        max_selections: int | None = None,
    ) -> None:
        super().__init__(message, default=default or [])
        self.min_selections = min_selections
        self.max_selections = max_selections

        # Convert string options to SelectOption
        self.options: list[SelectOption] = []
        for opt in options:
            if isinstance(opt, str):
                self.options.append(SelectOption(value=opt, label=opt))
            else:
                self.options.append(opt)

    def ask(self) -> PromptResponse[list[str]]:
        """Ask for multiple selections."""
        try:
            # Display options
            if RICH_AVAILABLE and self._console:
                self._console.print(f"\n[bold]{self.message}[/bold]")
                self._console.print("[dim]Enter comma-separated numbers or 'all'/'none'[/dim]")
                for i, opt in enumerate(self.options, 1):
                    disabled_marker = " [dim](disabled)[/dim]" if opt.disabled else ""
                    self._console.print(f"  {i}. {opt.label}{disabled_marker}")
            else:
                print(f"\n{self.message}")
                print("Enter comma-separated numbers or 'all'/'none'")
                for i, opt in enumerate(self.options, 1):
                    disabled_marker = " (disabled)" if opt.disabled else ""
                    print(f"  {i}. {opt.label}{disabled_marker}")

            response = input("Select: ").strip().lower()

            if response == "all":
                selections = [opt.value for opt in self.options if not opt.disabled]
            elif response == "none" or not response:
                selections = []
            else:
                selections = []
                for part in response.split(","):
                    part = part.strip()
                    try:
                        index = int(part) - 1
                        if 0 <= index < len(self.options):
                            opt = self.options[index]
                            if not opt.disabled:
                                selections.append(opt.value)
                    except ValueError:
                        continue

            # Validate selections
            if len(selections) < self.min_selections:
                print(f"Select at least {self.min_selections} options")
                return self.ask()

            if self.max_selections and len(selections) > self.max_selections:
                print(f"Select at most {self.max_selections} options")
                return self.ask()

            return PromptResponse(selections, PromptResult.ANSWERED)

        except (KeyboardInterrupt, EOFError):
            return PromptResponse([], PromptResult.CANCELLED)


# ========== Consent Prompt ==========


@dataclass
class ConsentInfo:
    """Information for consent request."""

    title: str
    description: str
    details: list[str] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)
    revocable: bool = True


class ConsentPrompt(Prompt[bool]):
    """Consent request with detailed explanation."""

    def __init__(
        self,
        consent_info: ConsentInfo,
        require_explicit: bool = True,
    ) -> None:
        super().__init__(consent_info.title)
        self.consent_info = consent_info
        self.require_explicit = require_explicit

    def ask(self) -> PromptResponse[bool]:
        """Ask for consent."""
        try:
            # Display consent information
            if RICH_AVAILABLE and self._console:
                self._display_rich()
            else:
                self._display_simple()

            # Get response
            if self.require_explicit:
                response = input("Type 'I AGREE' to consent (or 'no' to decline): ").strip()
                agreed = response.upper() == "I AGREE"
            else:
                confirm = ConfirmPrompt("Do you consent?", default=False)
                result = confirm.ask()
                agreed = result.value

            if not agreed:
                if RICH_AVAILABLE and self._console:
                    self._console.print("[yellow]Consent declined[/yellow]")
                else:
                    print("Consent declined")

            return PromptResponse(agreed, PromptResult.ANSWERED)

        except (KeyboardInterrupt, EOFError):
            return PromptResponse(False, PromptResult.CANCELLED)

    def _display_rich(self) -> None:
        """Display consent info with Rich formatting."""
        if not self._console:
            return

        content = Text()
        content.append(self.consent_info.description + "\n\n")

        if self.consent_info.details:
            content.append("Details:\n", style="bold")
            for detail in self.consent_info.details:
                content.append(f"  • {detail}\n")
            content.append("\n")

        if self.consent_info.implications:
            content.append("Implications:\n", style="bold yellow")
            for impl in self.consent_info.implications:
                content.append(f"  ⚠ {impl}\n", style="yellow")
            content.append("\n")

        if self.consent_info.revocable:
            content.append("This consent can be revoked at any time.\n", style="dim")

        panel = Panel(
            content,
            title=f"[bold]Consent Required: {self.consent_info.title}[/bold]",
            border_style="yellow",
        )
        self._console.print(panel)

    def _display_simple(self) -> None:
        """Display consent info with plain formatting."""
        print(f"\n{'=' * 60}")
        print(f"CONSENT REQUIRED: {self.consent_info.title}")
        print("=" * 60)
        print(self.consent_info.description)

        if self.consent_info.details:
            print("\nDetails:")
            for detail in self.consent_info.details:
                print(f"  • {detail}")

        if self.consent_info.implications:
            print("\nImplications:")
            for impl in self.consent_info.implications:
                print(f"  ⚠ {impl}")

        if self.consent_info.revocable:
            print("\nThis consent can be revoked at any time.")

        print("=" * 60)


# ========== Convenience Functions ==========


def confirm(
    message: str,
    default: bool = False,
    force: bool = False,
) -> bool:
    """Ask for confirmation.

    Args:
        message: The confirmation message
        default: Default value if user presses Enter
        force: If True, skip prompt and return True

    Returns:
        True if confirmed, False otherwise
    """
    if force:
        return True

    prompt = ConfirmPrompt(message, default=default)
    result = prompt.ask()
    return result.value if result.answered else False


def prompt_text(
    message: str,
    default: str | None = None,
    required: bool = False,
) -> str:
    """Ask for text input.

    Args:
        message: The prompt message
        default: Default value
        required: Whether input is required

    Returns:
        The user's input
    """
    prompt = TextPrompt(message, default=default, min_length=1 if required else 0)
    result = prompt.ask()

    if result.cancelled and required:
        raise typer.Abort()

    return result.value


def prompt_password(
    message: str = "Password",
    confirm: bool = False,
) -> str:
    """Ask for password input.

    Args:
        message: The prompt message
        confirm: Whether to require confirmation

    Returns:
        The password
    """
    prompt = PasswordPrompt(message, confirm=confirm)
    result = prompt.ask()

    if result.cancelled:
        raise typer.Abort()

    return result.value


def prompt_select(
    message: str,
    options: list[str] | list[SelectOption],
    default: str | None = None,
) -> str:
    """Ask for single selection.

    Args:
        message: The prompt message
        options: List of options
        default: Default selection

    Returns:
        The selected value
    """
    prompt = SelectPrompt(message, options=options, default=default)
    result = prompt.ask()

    if result.cancelled:
        raise typer.Abort()

    return result.value


def prompt_multi_select(
    message: str,
    options: list[str] | list[SelectOption],
    default: list[str] | None = None,
) -> list[str]:
    """Ask for multiple selection.

    Args:
        message: The prompt message
        options: List of options
        default: Default selections

    Returns:
        List of selected values
    """
    prompt = MultiSelectPrompt(message, options=options, default=default)
    result = prompt.ask()

    if result.cancelled:
        raise typer.Abort()

    return result.value


def request_consent(
    title: str,
    description: str,
    details: list[str] | None = None,
    implications: list[str] | None = None,
    force: bool = False,
) -> bool:
    """Request user consent.

    Args:
        title: Consent title
        description: Consent description
        details: List of details
        implications: List of implications
        force: If True, skip prompt and return True

    Returns:
        True if consent given, False otherwise
    """
    if force:
        return True

    info = ConsentInfo(
        title=title,
        description=description,
        details=details or [],
        implications=implications or [],
    )

    prompt = ConsentPrompt(info, require_explicit=True)
    result = prompt.ask()

    return result.value


# ========== Context-Aware Prompts ==========


def get_prompt_from_context(ctx: typer.Context) -> dict[str, Any]:
    """Get prompt configuration from Typer context."""
    obj = ctx.obj or {}
    return {
        "force": obj.get("force", False),
        "quiet": obj.get("quiet", False),
        "no_input": obj.get("no_input", False),
    }


def context_confirm(
    ctx: typer.Context,
    message: str,
    default: bool = False,
) -> bool:
    """Context-aware confirmation prompt."""
    config = get_prompt_from_context(ctx)

    if config["force"]:
        return True

    if config["quiet"] or config["no_input"]:
        return default

    return confirm(message, default=default)


def context_consent(
    ctx: typer.Context,
    title: str,
    description: str,
    **kwargs,
) -> bool:
    """Context-aware consent prompt."""
    config = get_prompt_from_context(ctx)

    if config["force"]:
        return True

    if config["quiet"] or config["no_input"]:
        return False

    return request_consent(title, description, **kwargs)
