"""
Example Exporter Plugins.

Demonstrates how to create exporter plugins for different output formats.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from proxima.plugins.base import ExporterPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class JSONExporterPlugin(ExporterPlugin):
    """Export results to JSON format.
    
    Features:
    - Pretty-printed output
    - Configurable indentation
    - Metadata inclusion
    - Timestamp addition
    
    Configuration:
        indent: Number of spaces for indentation (default: 2)
        include_metadata: Include export metadata (default: True)
        sort_keys: Sort dictionary keys (default: True)
    
    Example:
        plugin = JSONExporterPlugin({"indent": 4})
        plugin.export(results, "output.json")
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="json_exporter",
        version="1.0.0",
        plugin_type=PluginType.EXPORTER,
        description="Export results to JSON format with pretty printing",
        author="Proxima Team",
        provides=["json", "application/json"],
        config_schema={
            "type": "object",
            "properties": {
                "indent": {"type": "integer", "default": 2},
                "include_metadata": {"type": "boolean", "default": True},
                "sort_keys": {"type": "boolean", "default": True},
            },
        },
    )
    
    def get_format_name(self) -> str:
        """Return the export format name."""
        return "json"
    
    def export(self, data: Any, destination: str) -> None:
        """Export data to JSON file.
        
        Args:
            data: Data to export.
            destination: Output file path.
        """
        indent = self.get_config("indent", 2)
        include_metadata = self.get_config("include_metadata", True)
        sort_keys = self.get_config("sort_keys", True)
        
        output = data
        
        if include_metadata:
            output = {
                "_metadata": {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "exporter": self.name,
                    "version": self.version,
                },
                "data": data,
            }
        
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=indent, sort_keys=sort_keys, default=str)
        
        logger.info(f"Exported data to {destination}")
    
    def export_string(self, data: Any) -> str:
        """Export data to JSON string.
        
        Args:
            data: Data to export.
        
        Returns:
            JSON string.
        """
        indent = self.get_config("indent", 2)
        sort_keys = self.get_config("sort_keys", True)
        return json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)


class CSVExporterPlugin(ExporterPlugin):
    """Export results to CSV format.
    
    Handles nested dictionaries by flattening them.
    
    Configuration:
        delimiter: Field delimiter (default: ',')
        include_header: Include column headers (default: True)
        flatten_nested: Flatten nested dicts (default: True)
    
    Example:
        plugin = CSVExporterPlugin()
        plugin.export({"counts": {"00": 500, "11": 500}}, "output.csv")
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="csv_exporter",
        version="1.0.0",
        plugin_type=PluginType.EXPORTER,
        description="Export results to CSV format",
        author="Proxima Team",
        provides=["csv", "text/csv"],
        config_schema={
            "type": "object",
            "properties": {
                "delimiter": {"type": "string", "default": ","},
                "include_header": {"type": "boolean", "default": True},
                "flatten_nested": {"type": "boolean", "default": True},
            },
        },
    )
    
    def get_format_name(self) -> str:
        """Return the export format name."""
        return "csv"
    
    def export(self, data: Any, destination: str) -> None:
        """Export data to CSV file.
        
        Args:
            data: Data to export. Can be dict or list of dicts.
            destination: Output file path.
        """
        delimiter = self.get_config("delimiter", ",")
        include_header = self.get_config("include_header", True)
        flatten_nested = self.get_config("flatten_nested", True)
        
        # Convert to list of dicts
        if isinstance(data, dict):
            if flatten_nested:
                data = [self._flatten_dict(data)]
            else:
                data = [data]
        elif not isinstance(data, list):
            data = [{"value": data}]
        
        if not data:
            logger.warning("No data to export")
            return
        
        # Get all keys from all rows
        all_keys: set[str] = set()
        for row in data:
            if isinstance(row, dict):
                if flatten_nested:
                    row = self._flatten_dict(row)
                all_keys.update(row.keys())
        
        fieldnames = sorted(all_keys)
        
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            
            if include_header:
                writer.writeheader()
            
            for row in data:
                if isinstance(row, dict):
                    if flatten_nested:
                        row = self._flatten_dict(row)
                    writer.writerow(row)
        
        logger.info(f"Exported data to {destination}")
    
    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten.
            parent_key: Parent key prefix.
            sep: Key separator.
        
        Returns:
            Flattened dictionary.
        """
        items: list[tuple[str, Any]] = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}[{i}]", sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        
        return dict(items)


class MarkdownExporterPlugin(ExporterPlugin):
    """Export results to Markdown format.
    
    Creates readable Markdown reports with tables and formatting.
    
    Configuration:
        include_toc: Include table of contents (default: True)
        include_timestamp: Include export timestamp (default: True)
    
    Example:
        plugin = MarkdownExporterPlugin()
        plugin.export(results, "report.md")
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="markdown_exporter",
        version="1.0.0",
        plugin_type=PluginType.EXPORTER,
        description="Export results to Markdown format",
        author="Proxima Team",
        provides=["markdown", "md", "text/markdown"],
        config_schema={
            "type": "object",
            "properties": {
                "include_toc": {"type": "boolean", "default": True},
                "include_timestamp": {"type": "boolean", "default": True},
            },
        },
    )
    
    def get_format_name(self) -> str:
        """Return the export format name."""
        return "markdown"
    
    def export(self, data: Any, destination: str) -> None:
        """Export data to Markdown file.
        
        Args:
            data: Data to export.
            destination: Output file path.
        """
        include_toc = self.get_config("include_toc", True)
        include_timestamp = self.get_config("include_timestamp", True)
        
        lines = ["# Proxima Execution Report", ""]
        
        if include_timestamp:
            lines.extend([
                f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
                "",
            ])
        
        if include_toc:
            lines.extend([
                "## Table of Contents",
                "",
                "- [Summary](#summary)",
                "- [Results](#results)",
                "- [Details](#details)",
                "",
            ])
        
        lines.extend(["## Summary", ""])
        
        # Generate summary section
        if isinstance(data, dict):
            if "backend" in data:
                lines.append(f"**Backend:** {data['backend']}")
            if "execution_time_ms" in data:
                lines.append(f"**Execution Time:** {data['execution_time_ms']:.2f} ms")
            if "shots" in data:
                lines.append(f"**Shots:** {data['shots']}")
            lines.append("")
        
        lines.extend(["## Results", ""])
        
        # Generate results table
        if isinstance(data, dict) and "counts" in data:
            counts = data["counts"]
            lines.extend([
                "| State | Count | Probability |",
                "|-------|-------|-------------|",
            ])
            
            total = sum(counts.values())
            for state, count in sorted(counts.items()):
                prob = count / total if total > 0 else 0
                lines.append(f"| `{state}` | {count} | {prob:.4f} |")
            lines.append("")
        
        lines.extend(["## Details", ""])
        
        # Add raw data as code block
        lines.extend([
            "```json",
            json.dumps(data, indent=2, default=str),
            "```",
            "",
        ])
        
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Exported data to {destination}")
