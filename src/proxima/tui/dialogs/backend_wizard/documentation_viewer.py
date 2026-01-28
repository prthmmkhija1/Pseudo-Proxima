"""Documentation Viewer.

Screen for viewing generated backend documentation.
Renders markdown content with syntax highlighting.

Part of Phase 8: Final Deployment & Success Confirmation.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import Static, Button, Label
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel


class DocumentationViewer(ModalScreen):
    """Viewer for generated backend documentation.
    
    Renders markdown content with:
    - Syntax highlighting for code blocks
    - Table formatting
    - Headers and sections
    - Copy to clipboard functionality
    """
    
    DEFAULT_CSS = """
    DocumentationViewer {
        align: center middle;
    }
    
    DocumentationViewer #main_container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $primary;
    }
    
    DocumentationViewer .header {
        width: 100%;
        height: auto;
        padding: 1;
        background: $primary-darken-2;
    }
    
    DocumentationViewer .header-title {
        text-style: bold;
        color: $text;
    }
    
    DocumentationViewer .toolbar {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: $surface-darken-1;
        border-bottom: solid $primary-darken-3;
    }
    
    DocumentationViewer .toolbar Button {
        margin: 0 1;
        min-width: 10;
    }
    
    DocumentationViewer .content {
        height: 1fr;
        padding: 1 2;
    }
    
    DocumentationViewer .markdown-content {
        width: 100%;
        height: auto;
    }
    
    DocumentationViewer .footer {
        width: 100%;
        height: auto;
        padding: 1;
        align: center middle;
        border-top: solid $primary-darken-3;
    }
    
    DocumentationViewer .toc-item {
        padding: 0 1;
        color: $primary;
    }
    
    DocumentationViewer .toc-item:hover {
        background: $primary 20%;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("c", "copy", "Copy"),
        ("e", "export", "Export"),
        ("up", "scroll_up", "Scroll Up"),
        ("down", "scroll_down", "Scroll Down"),
        ("home", "scroll_home", "Scroll to Top"),
        ("end", "scroll_end", "Scroll to Bottom"),
    ]
    
    def __init__(
        self,
        content: str,
        backend_name: str = "Backend",
        file_path: Optional[Path] = None,
        **kwargs
    ):
        """Initialize documentation viewer.
        
        Args:
            content: Markdown content to display
            backend_name: Name of the backend
            file_path: Optional path to the documentation file
        """
        super().__init__(**kwargs)
        self.content = content
        self.backend_name = backend_name
        self.file_path = file_path
        
        # Parse table of contents
        self.toc = self._parse_toc()
    
    def _parse_toc(self) -> list:
        """Parse table of contents from markdown headers."""
        toc = []
        
        for line in self.content.split("\n"):
            if line.startswith("## "):
                title = line[3:].strip()
                toc.append({"level": 2, "title": title})
            elif line.startswith("### "):
                title = line[4:].strip()
                toc.append({"level": 3, "title": title})
        
        return toc
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="main_container"):
            # Header
            with Horizontal(classes="header"):
                yield Static(
                    f"📖 Documentation: {self.backend_name}",
                    classes="header-title"
                )
            
            # Toolbar
            with Horizontal(classes="toolbar"):
                yield Button("📋 Copy All", id="copy", variant="default")
                yield Button("💾 Export", id="export", variant="default")
                yield Button("🔍 Find", id="find", variant="default")
            
            # Main content area
            with Horizontal():
                # Table of contents sidebar (if there are headers)
                if self.toc:
                    with Vertical(id="toc_sidebar"):
                        yield Static("Contents", classes="toc-title")
                        for item in self.toc[:10]:  # Show first 10
                            indent = "  " if item["level"] == 3 else ""
                            yield Static(
                                f"{indent}• {item['title'][:25]}",
                                classes="toc-item"
                            )
                
                # Content
                with ScrollableContainer(classes="content", id="docs_scroll"):
                    # Render markdown
                    md = Markdown(self.content)
                    yield Static(md, classes="markdown-content", id="markdown")
            
            # Footer
            with Horizontal(classes="footer"):
                if self.file_path:
                    yield Static(f"File: {self.file_path}", classes="file-path")
                yield Button("Close", id="close", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "close":
            self.dismiss()
        
        elif button_id == "copy":
            self._copy_to_clipboard()
        
        elif button_id == "export":
            self._export_documentation()
        
        elif button_id == "find":
            self._show_find_dialog()
    
    def _copy_to_clipboard(self) -> None:
        """Copy documentation to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(self.content)
            self.notify("Documentation copied to clipboard!")
        except ImportError:
            # Fallback: show message
            self.notify(
                "Install pyperclip for clipboard support: pip install pyperclip",
                severity="warning"
            )
        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")
    
    def _export_documentation(self) -> None:
        """Export documentation to file."""
        from datetime import datetime
        
        # Generate filename
        safe_name = self.backend_name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_docs_{timestamp}.md"
        
        try:
            output_path = Path.home() / filename
            output_path.write_text(self.content)
            self.notify(f"Documentation exported to: {output_path}")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _show_find_dialog(self) -> None:
        """Show find dialog."""
        self.notify("Use Ctrl+F in your terminal for search", severity="information")
    
    def action_close(self) -> None:
        """Handle close action."""
        self.dismiss()
    
    def action_copy(self) -> None:
        """Handle copy action."""
        self._copy_to_clipboard()
    
    def action_export(self) -> None:
        """Handle export action."""
        self._export_documentation()
    
    def action_scroll_up(self) -> None:
        """Scroll up."""
        scroll = self.query_one("#docs_scroll", ScrollableContainer)
        scroll.scroll_up()
    
    def action_scroll_down(self) -> None:
        """Scroll down."""
        scroll = self.query_one("#docs_scroll", ScrollableContainer)
        scroll.scroll_down()
    
    def action_scroll_home(self) -> None:
        """Scroll to top."""
        scroll = self.query_one("#docs_scroll", ScrollableContainer)
        scroll.scroll_home()
    
    def action_scroll_end(self) -> None:
        """Scroll to bottom."""
        scroll = self.query_one("#docs_scroll", ScrollableContainer)
        scroll.scroll_end()


class CodeDocumentationViewer(ModalScreen):
    """Viewer specifically for code documentation with syntax highlighting."""
    
    DEFAULT_CSS = """
    CodeDocumentationViewer {
        align: center middle;
    }
    
    CodeDocumentationViewer #main_container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $primary;
    }
    
    CodeDocumentationViewer .header {
        width: 100%;
        height: auto;
        padding: 1;
        background: $primary-darken-2;
    }
    
    CodeDocumentationViewer .content {
        height: 1fr;
        padding: 1;
    }
    
    CodeDocumentationViewer .code-block {
        background: $surface-darken-2;
        padding: 1;
        font-family: monospace;
    }
    
    CodeDocumentationViewer .footer {
        width: 100%;
        height: auto;
        padding: 1;
        align: center middle;
        border-top: solid $primary-darken-3;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
    ]
    
    def __init__(
        self,
        code: str,
        language: str = "python",
        title: str = "Code",
        **kwargs
    ):
        """Initialize code viewer.
        
        Args:
            code: Code content
            language: Programming language for syntax highlighting
            title: Title for the viewer
        """
        super().__init__(**kwargs)
        self.code = code
        self.language = language
        self.title = title
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="main_container"):
            # Header
            with Horizontal(classes="header"):
                yield Static(f"📝 {self.title}", classes="header-title")
            
            # Content with syntax highlighting
            with ScrollableContainer(classes="content"):
                syntax = Syntax(
                    self.code,
                    self.language,
                    theme="monokai",
                    line_numbers=True,
                )
                yield Static(syntax, classes="code-block")
            
            # Footer
            with Horizontal(classes="footer"):
                yield Button("📋 Copy", id="copy", variant="default")
                yield Button("Close", id="close", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close":
            self.dismiss()
        elif event.button.id == "copy":
            self._copy_code()
    
    def _copy_code(self) -> None:
        """Copy code to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(self.code)
            self.notify("Code copied to clipboard!")
        except ImportError:
            self.notify(
                "Install pyperclip for clipboard support",
                severity="warning"
            )
    
    def action_close(self) -> None:
        """Handle close action."""
        self.dismiss()
