"""TUI Styles and Theming for Proxima.

Step 6.1: Comprehensive styling system with:
- Color palettes (dark/light themes)
- CSS component classes
- Animation definitions
- Responsive breakpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class Theme(Enum):
    """Available TUI themes."""

    DARK = "dark"
    LIGHT = "light"
    OCEAN = "ocean"
    FOREST = "forest"
    SUNSET = "sunset"


@dataclass
class ColorPalette:
    """Color palette definition."""

    # Primary colors
    primary: str = "#3498db"
    primary_light: str = "#5dade2"
    primary_dark: str = "#2980b9"

    # Secondary colors
    secondary: str = "#2ecc71"
    secondary_light: str = "#58d68d"
    secondary_dark: str = "#27ae60"

    # Background colors
    background: str = "#1a1a2e"
    surface: str = "#16213e"
    surface_light: str = "#1f3460"
    surface_lighter: str = "#2a4a7c"

    # Text colors
    text: str = "#ecf0f1"
    text_muted: str = "#95a5a6"
    text_disabled: str = "#5d6d7e"

    # Status colors
    success: str = "#2ecc71"
    warning: str = "#f39c12"
    error: str = "#e74c3c"
    info: str = "#3498db"

    # Accent colors
    accent: str = "#9b59b6"
    highlight: str = "#f1c40f"

    def to_css_vars(self) -> str:
        """Convert palette to CSS variable definitions."""
        return f"""
$primary: {self.primary};
$primary-light: {self.primary_light};
$primary-dark: {self.primary_dark};
$secondary: {self.secondary};
$secondary-light: {self.secondary_light};
$secondary-dark: {self.secondary_dark};
$background: {self.background};
$surface: {self.surface};
$surface-light: {self.surface_light};
$surface-lighter: {self.surface_lighter};
$text: {self.text};
$text-muted: {self.text_muted};
$text-disabled: {self.text_disabled};
$success: {self.success};
$warning: {self.warning};
$error: {self.error};
$info: {self.info};
$accent: {self.accent};
$highlight: {self.highlight};
"""


# Predefined color palettes
DARK_PALETTE = ColorPalette()

LIGHT_PALETTE = ColorPalette(
    primary="#2980b9",
    primary_light="#3498db",
    primary_dark="#1a5276",
    secondary="#27ae60",
    secondary_light="#2ecc71",
    secondary_dark="#1e8449",
    background="#f8f9fa",
    surface="#ffffff",
    surface_light="#ecf0f1",
    surface_lighter="#d5dbdb",
    text="#2c3e50",
    text_muted="#7f8c8d",
    text_disabled="#bdc3c7",
    success="#27ae60",
    warning="#e67e22",
    error="#c0392b",
    info="#2980b9",
    accent="#8e44ad",
    highlight="#f39c12",
)

OCEAN_PALETTE = ColorPalette(
    primary="#0077b6",
    primary_light="#00a8e8",
    primary_dark="#023e8a",
    secondary="#00b4d8",
    secondary_light="#48cae4",
    secondary_dark="#0096c7",
    background="#03045e",
    surface="#023e8a",
    surface_light="#0077b6",
    surface_lighter="#0096c7",
    text="#caf0f8",
    text_muted="#90e0ef",
    text_disabled="#48cae4",
)

FOREST_PALETTE = ColorPalette(
    primary="#2d6a4f",
    primary_light="#40916c",
    primary_dark="#1b4332",
    secondary="#52b788",
    secondary_light="#74c69d",
    secondary_dark="#40916c",
    background="#1b4332",
    surface="#2d6a4f",
    surface_light="#40916c",
    surface_lighter="#52b788",
    text="#d8f3dc",
    text_muted="#b7e4c7",
    text_disabled="#95d5b2",
)

SUNSET_PALETTE = ColorPalette(
    primary="#e63946",
    primary_light="#f07167",
    primary_dark="#c1121f",
    secondary="#ff6b35",
    secondary_light="#ff8c42",
    secondary_dark="#e55812",
    background="#1d3557",
    surface="#264653",
    surface_light="#2a9d8f",
    surface_lighter="#e9c46a",
    text="#f4f1de",
    text_muted="#e9c46a",
    text_disabled="#a8dadc",
)

PALETTES: dict[Theme, ColorPalette] = {
    Theme.DARK: DARK_PALETTE,
    Theme.LIGHT: LIGHT_PALETTE,
    Theme.OCEAN: OCEAN_PALETTE,
    Theme.FOREST: FOREST_PALETTE,
    Theme.SUNSET: SUNSET_PALETTE,
}


def get_palette(theme: Theme | str) -> ColorPalette:
    """Get the color palette for a theme.

    Args:
        theme: Theme enum or string name

    Returns:
        ColorPalette for the theme
    """
    if isinstance(theme, str):
        theme = Theme(theme)
    return PALETTES.get(theme, DARK_PALETTE)


# ========== CSS Component Styles ==========


@dataclass
class CSSComponent:
    """Base CSS component definition."""

    name: str
    css: str
    description: str = ""


# Base screen styles
SCREEN_CSS = CSSComponent(
    name="screen",
    description="Base screen styles",
    css="""
Screen {
    background: $background;
}

Screen > Container {
    height: 100%;
    width: 100%;
}
""",
)

# Header and footer styles
HEADER_FOOTER_CSS = CSSComponent(
    name="header_footer",
    description="Header and footer styles",
    css="""
Header {
    background: $primary;
    color: $text;
    text-style: bold;
}

Footer {
    background: $surface;
    color: $text;
}

.footer-key {
    color: $primary;
    text-style: bold;
}
""",
)

# Button styles
BUTTON_CSS = CSSComponent(
    name="buttons",
    description="Button component styles",
    css="""
Button {
    margin: 0 1;
    min-width: 12;
}

Button:hover {
    background: $primary-light;
}

Button:focus {
    border: heavy $primary;
}

Button.-primary {
    background: $primary;
}

Button.-primary:hover {
    background: $primary-light;
}

Button.-success {
    background: $success;
}

Button.-success:hover {
    background: $secondary-light;
}

Button.-warning {
    background: $warning;
}

Button.-error {
    background: $error;
}

Button.-disabled {
    background: $surface;
    color: $text-disabled;
}
""",
)

# Input styles
INPUT_CSS = CSSComponent(
    name="inputs",
    description="Input field styles",
    css="""
Input {
    background: $surface;
    border: solid $primary;
    padding: 0 1;
}

Input:focus {
    border: heavy $primary-light;
}

Input.-invalid {
    border: solid $error;
}

Input::placeholder {
    color: $text-muted;
}

Switch {
    background: $surface;
}

Switch:focus {
    border: heavy $primary;
}

Switch.-on {
    background: $success;
}
""",
)

# Container styles
CONTAINER_CSS = CSSComponent(
    name="containers",
    description="Container and layout styles",
    css="""
.panel {
    border: solid $primary;
    padding: 1;
    margin: 1;
}

.panel-title {
    text-style: bold;
    padding: 1;
    background: $surface;
    color: $text;
}

.card {
    border: solid $surface-light;
    padding: 1;
    margin: 1;
    background: $surface;
}

.card:hover {
    border: solid $primary;
}

.card:focus {
    border: heavy $primary;
}

.section {
    margin: 1;
    padding: 1;
}

.section-title {
    text-style: bold;
    padding-bottom: 1;
    border-bottom: solid $surface-light;
}

ScrollableContainer {
    scrollbar-color: $primary;
    scrollbar-background: $surface;
}
""",
)

# Table styles
TABLE_CSS = CSSComponent(
    name="tables",
    description="Data table styles",
    css="""
DataTable {
    background: $surface;
    border: solid $primary;
}

DataTable > .datatable--header {
    background: $primary;
    color: $text;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: $primary-dark;
    color: $text;
}

DataTable > .datatable--hover {
    background: $surface-light;
}

DataTable:focus > .datatable--cursor {
    background: $primary;
}
""",
)

# Status indicator styles
STATUS_CSS = CSSComponent(
    name="status",
    description="Status indicator styles",
    css="""
.status-success {
    color: $success;
}

.status-warning {
    color: $warning;
}

.status-error {
    color: $error;
}

.status-info {
    color: $info;
}

.status-pending {
    color: $text-muted;
}

.status-running {
    color: $accent;
}

.badge {
    padding: 0 1;
    background: $surface;
    border: solid $primary;
}

.badge-success {
    background: $success;
    color: $text;
}

.badge-warning {
    background: $warning;
    color: $background;
}

.badge-error {
    background: $error;
    color: $text;
}
""",
)

# Progress bar styles
PROGRESS_CSS = CSSComponent(
    name="progress",
    description="Progress bar and indicator styles",
    css="""
ProgressBar {
    padding: 0 1;
}

ProgressBar > .bar--bar {
    color: $primary;
}

ProgressBar > .bar--complete {
    color: $success;
}

ProgressBar > .bar--background {
    color: $surface;
}

.progress-label {
    text-align: center;
    color: $text-muted;
}

.spinner {
    color: $primary;
}
""",
)

# Modal and dialog styles
MODAL_CSS = CSSComponent(
    name="modals",
    description="Modal and dialog styles",
    css="""
.modal-container {
    align: center middle;
    width: 100%;
    height: 100%;
}

.modal {
    width: auto;
    max-width: 80%;
    height: auto;
    max-height: 80%;
    border: thick $primary;
    background: $surface;
    padding: 2;
}

.modal-title {
    text-align: center;
    text-style: bold;
    margin-bottom: 2;
    color: $primary;
}

.modal-content {
    margin: 1 0;
}

.modal-buttons {
    margin-top: 2;
    align: center middle;
}

.modal-buttons Button {
    margin: 0 2;
}
""",
)

# Log viewer styles
LOG_CSS = CSSComponent(
    name="logs",
    description="Log viewer styles",
    css="""
.log-container {
    background: $background;
    border: solid $surface;
    padding: 1;
}

.log-entry {
    padding: 0 1;
}

.log-timestamp {
    color: $text-muted;
}

.log-level-debug {
    color: $text-muted;
}

.log-level-info {
    color: $info;
}

.log-level-warning {
    color: $warning;
}

.log-level-error {
    color: $error;
}

.log-message {
    color: $text;
}
""",
)

# Toast notification styles
TOAST_CSS = CSSComponent(
    name="toasts",
    description="Toast notification styles",
    css="""
Toast {
    background: $surface;
    border: solid $primary;
    padding: 1;
}

Toast.-information {
    border: solid $info;
}

Toast.-warning {
    border: solid $warning;
}

Toast.-error {
    border: solid $error;
}

Toast.-success {
    border: solid $success;
}
""",
)

# Tree view styles
TREE_CSS = CSSComponent(
    name="trees",
    description="Tree view styles",
    css="""
Tree {
    background: $surface;
    padding: 1;
}

Tree > .tree--label {
    color: $text;
}

Tree > .tree--cursor {
    background: $primary-dark;
    color: $text;
}

Tree > .tree--highlight {
    background: $surface-light;
}

Tree:focus > .tree--cursor {
    background: $primary;
}

.tree-folder {
    color: $primary;
}

.tree-file {
    color: $text;
}
""",
)


# ========== Complete Theme CSS ==========


def build_theme_css(palette: ColorPalette | None = None) -> str:
    """Build complete theme CSS with all components.

    Args:
        palette: Optional color palette (defaults to dark)

    Returns:
        Complete CSS string
    """
    if palette is None:
        palette = DARK_PALETTE

    # Collect all component CSS
    components = [
        SCREEN_CSS,
        HEADER_FOOTER_CSS,
        BUTTON_CSS,
        INPUT_CSS,
        CONTAINER_CSS,
        TABLE_CSS,
        STATUS_CSS,
        PROGRESS_CSS,
        MODAL_CSS,
        LOG_CSS,
        TOAST_CSS,
        TREE_CSS,
    ]

    css_parts = [
        "/* Proxima TUI Theme */",
        "/* Auto-generated - Do not edit directly */",
        "",
        palette.to_css_vars(),
    ]

    for component in components:
        css_parts.append(f"\n/* {component.name} - {component.description} */")
        css_parts.append(component.css)

    return "\n".join(css_parts)


# ========== Animation Definitions ==========


@dataclass
class Animation:
    """CSS animation definition."""

    name: str
    keyframes: str
    duration: str = "1s"
    timing: str = "ease-in-out"
    iteration: str = "infinite"

    def to_css(self) -> str:
        """Convert to CSS animation definition."""
        return f"""
@keyframes {self.name} {{
{self.keyframes}
}}

.animate-{self.name} {{
    animation: {self.name} {self.duration} {self.timing} {self.iteration};
}}
"""


PULSE_ANIMATION = Animation(
    name="pulse",
    keyframes="""
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
""",
    duration="2s",
)

SPIN_ANIMATION = Animation(
    name="spin",
    keyframes="""
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
""",
    duration="1s",
)

FADE_IN_ANIMATION = Animation(
    name="fade-in",
    keyframes="""
    0% { opacity: 0; }
    100% { opacity: 1; }
""",
    duration="0.3s",
    iteration="1",
)


# ========== Style Manager ==========


class StyleManager:
    """Manages TUI styles and theming."""

    _instance: ClassVar[StyleManager | None] = None
    _theme: Theme = Theme.DARK
    _palette: ColorPalette = DARK_PALETTE
    _css_cache: str = ""

    def __new__(cls) -> StyleManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build_css()
        return cls._instance

    def _build_css(self) -> None:
        """Build and cache the CSS."""
        self._css_cache = build_theme_css(self._palette)

    @property
    def theme(self) -> Theme:
        """Get current theme."""
        return self._theme

    @theme.setter
    def theme(self, value: Theme | str) -> None:
        """Set current theme."""
        if isinstance(value, str):
            value = Theme(value)
        self._theme = value
        self._palette = PALETTES.get(value, DARK_PALETTE)
        self._build_css()

    @property
    def palette(self) -> ColorPalette:
        """Get current color palette."""
        return self._palette

    @property
    def css(self) -> str:
        """Get current CSS."""
        return self._css_cache

    def get_status_color(self, status: str) -> str:
        """Get color for a status value.

        Args:
            status: Status string (success, error, warning, etc.)

        Returns:
            CSS color value
        """
        status_colors = {
            "success": self._palette.success,
            "completed": self._palette.success,
            "error": self._palette.error,
            "failed": self._palette.error,
            "warning": self._palette.warning,
            "pending": self._palette.text_muted,
            "running": self._palette.accent,
            "info": self._palette.info,
        }
        return status_colors.get(status.lower(), self._palette.text)


# Global style manager instance
style_manager = StyleManager()


def get_css() -> str:
    """Get the current theme CSS."""
    return style_manager.css


def set_theme(theme: Theme | str) -> None:
    """Set the current theme.

    Args:
        theme: Theme enum or string name
    """
    style_manager.theme = theme
