"""Code Preview Dialog.

Modal dialog for previewing generated backend code with syntax highlighting.
Supports multiple files with tabbed navigation.
"""

from __future__ import annotations

from typing import Dict, Optional

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, TabbedContent, TabPane, TextArea
from textual.screen import ModalScreen


class CodePreviewDialog(ModalScreen[bool]):
    """Dialog to preview generated code files.
    
    Shows generated code with syntax highlighting and allows
    users to review before deployment.
    """
    
    DEFAULT_CSS = """
    CodePreviewDialog {
        align: center middle;
    }
    
    CodePreviewDialog .dialog-container {
        width: 95%;
        height: 90%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    CodePreviewDialog .dialog-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    CodePreviewDialog .code-tabs {
        height: 1fr;
        margin: 1 0;
    }
    
    CodePreviewDialog .code-area {
        height: 100%;
        border: solid $primary-darken-3;
    }
    
    CodePreviewDialog .stats-bar {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border-top: solid $primary-darken-3;
    }
    
    CodePreviewDialog .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    CodePreviewDialog .dialog-button {
        margin: 0 1;
        min-width: 14;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+c", "copy", "Copy"),
    ]
    
    def __init__(self, generated_code: Dict[str, str], title: str = "Generated Code Preview"):
        """Initialize the code preview dialog.
        
        Args:
            generated_code: Dictionary mapping file names to code content
            title: Dialog title
        """
        super().__init__()
        self.generated_code = generated_code
        self.title_text = title
    
    def compose(self) -> ComposeResult:
        """Create dialog widgets."""
        with Center():
            with Vertical(classes="dialog-container"):
                # Title
                yield Static(
                    f"📄 {self.title_text}",
                    classes="dialog-title"
                )
                
                # Tabbed content for each file
                with TabbedContent(classes="code-tabs"):
                    for file_name, code in self.generated_code.items():
                        # Get just the filename for the tab
                        short_name = file_name.split('/')[-1]
                        
                        with TabPane(short_name, id=f"tab_{self._safe_id(file_name)}"):
                            yield Static(
                                f"📁 {file_name}",
                                classes="file-path"
                            )
                            yield TextArea(
                                code,
                                language="python" if file_name.endswith('.py') else "markdown",
                                read_only=True,
                                show_line_numbers=True,
                                classes="code-area"
                            )
                
                # Stats bar
                total_lines = sum(len(code.split('\n')) for code in self.generated_code.values())
                total_files = len(self.generated_code)
                
                yield Static(
                    f"📊 {total_files} files | {total_lines} total lines | "
                    f"Use Ctrl+C to copy selected code",
                    classes="stats-bar"
                )
                
                # Buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "Copy All",
                        id="btn_copy_all",
                        variant="default",
                        classes="dialog-button"
                    )
                    yield Button(
                        "Close",
                        id="btn_close",
                        variant="primary",
                        classes="dialog-button"
                    )
    
    def _safe_id(self, file_name: str) -> str:
        """Convert file name to safe ID."""
        return file_name.replace('/', '_').replace('.', '_').replace('-', '_')
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_close":
            self.dismiss(True)
        
        elif event.button.id == "btn_copy_all":
            self._copy_all_code()
    
    def _copy_all_code(self) -> None:
        """Copy all generated code to clipboard."""
        all_code = []
        
        for file_name, code in self.generated_code.items():
            all_code.append(f"# {'=' * 60}")
            all_code.append(f"# File: {file_name}")
            all_code.append(f"# {'=' * 60}")
            all_code.append(code)
            all_code.append("")
        
        combined = '\n'.join(all_code)
        
        # Try to copy to clipboard
        try:
            import pyperclip
            pyperclip.copy(combined)
            self.notify("Code copied to clipboard!", severity="information")
        except ImportError:
            self.notify(
                "Install pyperclip for clipboard support: pip install pyperclip",
                severity="warning"
            )
    
    def action_close(self) -> None:
        """Handle escape key."""
        self.dismiss(True)
    
    def action_copy(self) -> None:
        """Copy currently selected code."""
        self._copy_all_code()


class SingleFilePreviewDialog(ModalScreen[bool]):
    """Dialog to preview a single file's code."""
    
    DEFAULT_CSS = """
    SingleFilePreviewDialog {
        align: center middle;
    }
    
    SingleFilePreviewDialog .dialog-container {
        width: 90%;
        height: 85%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    SingleFilePreviewDialog .dialog-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
    }
    
    SingleFilePreviewDialog .code-area {
        height: 1fr;
        margin: 1 0;
        border: solid $primary-darken-3;
    }
    
    SingleFilePreviewDialog .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }
    """
    
    def __init__(self, file_name: str, code: str):
        """Initialize single file preview.
        
        Args:
            file_name: Name of the file
            code: File content
        """
        super().__init__()
        self.file_name = file_name
        self.code = code
    
    def compose(self) -> ComposeResult:
        """Create dialog widgets."""
        with Center():
            with Vertical(classes="dialog-container"):
                yield Static(
                    f"📄 {self.file_name}",
                    classes="dialog-title"
                )
                
                # Determine language from file extension
                language = "python"
                if self.file_name.endswith('.md'):
                    language = "markdown"
                elif self.file_name.endswith('.yaml') or self.file_name.endswith('.yml'):
                    language = "yaml"
                elif self.file_name.endswith('.json'):
                    language = "json"
                
                yield TextArea(
                    self.code,
                    language=language,
                    read_only=True,
                    show_line_numbers=True,
                    classes="code-area"
                )
                
                with Horizontal(classes="button-container"):
                    yield Button(
                        "Close",
                        id="btn_close",
                        variant="primary"
                    )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "btn_close":
            self.dismiss(True)
