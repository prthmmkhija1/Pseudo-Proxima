"""Visualization tools for LRET vs Cirq benchmark results.

This module provides plotting and report generation for
LRET Cirq Scalability benchmark data. Features:
- 4-panel comparison plots (speedup, time, fidelity, rank)
- Interactive plot generation for TUI
- Markdown report generation
- CSV data loading and processing

Requires: matplotlib>=3.5, pandas>=1.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from proxima.backends.lret.cirq_scalability import BenchmarkResult, CirqScalabilityMetrics

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for benchmark plots.
    
    Attributes:
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch
        style: Matplotlib style ('default', 'seaborn', 'ggplot', etc.)
        colors: Color scheme for plots
        title_fontsize: Font size for titles
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legends
        save_format: Output format ('png', 'pdf', 'svg')
    """
    
    figsize: tuple[float, float] = (12, 10)
    dpi: int = 100
    style: str = 'default'
    colors: dict[str, str] = None
    title_fontsize: int = 12
    label_fontsize: int = 10
    legend_fontsize: int = 9
    save_format: str = 'png'
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'lret': '#2ecc71',      # Green
                'cirq': '#3498db',      # Blue
                'speedup': '#e74c3c',   # Red
                'fidelity': '#9b59b6',  # Purple
                'rank': '#f39c12',      # Orange
                'grid': '#bdc3c7',      # Gray
            }


def load_benchmark_csv(csv_path: Union[str, Path]) -> BenchmarkResult:
    """Load benchmark data from CSV file.
    
    Args:
        csv_path: Path to CSV file with benchmark data
        
    Returns:
        BenchmarkResult populated from CSV
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for CSV loading. Install with: pip install pandas")
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    metrics = []
    for _, row in df.iterrows():
        metrics.append(CirqScalabilityMetrics(
            lret_time_ms=row.get('lret_time_ms', 0),
            cirq_fdm_time_ms=row.get('cirq_fdm_time_ms', 0),
            speedup_factor=row.get('speedup', 0),
            lret_final_rank=int(row.get('final_rank', 0)),
            fidelity=row.get('fidelity', 0),
            trace_distance=row.get('trace_distance', 0),
            qubit_count=int(row.get('qubits', 0)),
            circuit_depth=int(row.get('depth', 0)),
        ))
    
    result = BenchmarkResult(metrics=metrics)
    result.csv_path = csv_path
    result.compute_summary()
    
    return result


def plot_lret_cirq_comparison(
    benchmark_result: BenchmarkResult,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    show: bool = False,
) -> Path:
    """Generate 4-panel comparison plot for LRET vs Cirq benchmarks.
    
    Creates a 2x2 grid of plots:
    - Top-left: Speedup vs Qubits
    - Top-right: Execution Time Comparison
    - Bottom-left: Fidelity vs Qubits  
    - Bottom-right: Rank Growth vs Qubits
    
    Args:
        benchmark_result: BenchmarkResult with metrics to plot
        output_path: Path to save plot (auto-generated if None)
        config: PlotConfig for styling
        show: Whether to display plot interactively
        
    Returns:
        Path to saved plot file
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')  # Non-interactive backend for file output
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    config = config or PlotConfig()
    
    # Extract data
    metrics = benchmark_result.metrics
    if not metrics:
        raise ValueError("No benchmark metrics to plot")
    
    qubits = [m.qubit_count for m in metrics]
    lret_times = [m.lret_time_ms for m in metrics]
    cirq_times = [m.cirq_fdm_time_ms for m in metrics]
    speedups = [m.speedup_factor for m in metrics]
    fidelities = [m.fidelity for m in metrics]
    ranks = [m.lret_final_rank for m in metrics]
    
    # Apply style
    if config.style != 'default':
        try:
            plt.style.use(config.style)
        except Exception:
            pass
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=config.figsize, dpi=config.dpi)
    fig.suptitle(
        'LRET vs Cirq FDM Performance Comparison',
        fontsize=config.title_fontsize + 2,
        fontweight='bold',
        y=0.98,
    )
    
    # =========================================================================
    # Plot 1: Speedup vs Qubits (Top-Left)
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.plot(qubits, speedups, 'o-', 
             color=config.colors['speedup'], 
             linewidth=2, 
             markersize=8,
             label='LRET Speedup')
    ax1.axhline(y=1.0, color=config.colors['grid'], linestyle='--', linewidth=1, alpha=0.7)
    ax1.fill_between(qubits, 1, speedups, 
                     where=[s > 1 for s in speedups],
                     color=config.colors['speedup'], 
                     alpha=0.2)
    ax1.set_xlabel('Number of Qubits', fontsize=config.label_fontsize)
    ax1.set_ylabel('Speedup Factor (x)', fontsize=config.label_fontsize)
    ax1.set_title('Speedup vs Qubit Count', fontsize=config.title_fontsize)
    ax1.legend(fontsize=config.legend_fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(qubits)
    
    # Annotate max speedup
    max_idx = speedups.index(max(speedups))
    ax1.annotate(
        f'{speedups[max_idx]:.1f}x',
        xy=(qubits[max_idx], speedups[max_idx]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=config.legend_fontsize,
        fontweight='bold',
    )
    
    # =========================================================================
    # Plot 2: Execution Time Comparison (Top-Right)
    # =========================================================================
    ax2 = axes[0, 1]
    width = 0.35
    x_pos = np.arange(len(qubits))
    
    bars1 = ax2.bar(x_pos - width/2, lret_times, width, 
                    label='LRET', 
                    color=config.colors['lret'],
                    alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, cirq_times, width, 
                    label='Cirq FDM', 
                    color=config.colors['cirq'],
                    alpha=0.8)
    
    ax2.set_xlabel('Number of Qubits', fontsize=config.label_fontsize)
    ax2.set_ylabel('Execution Time (ms)', fontsize=config.label_fontsize)
    ax2.set_title('Execution Time Comparison', fontsize=config.title_fontsize)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(qubits)
    ax2.legend(fontsize=config.legend_fontsize)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Use log scale if times vary greatly
    if max(cirq_times) / (min(lret_times) + 0.001) > 100:
        ax2.set_yscale('log')
    
    # =========================================================================
    # Plot 3: Fidelity vs Qubits (Bottom-Left)
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.plot(qubits, fidelities, 's-', 
             color=config.colors['fidelity'], 
             linewidth=2, 
             markersize=8,
             label='State Fidelity')
    ax3.axhline(y=0.99, color=config.colors['grid'], linestyle='--', 
                linewidth=1, alpha=0.7, label='99% Threshold')
    ax3.fill_between(qubits, 0, fidelities, 
                     color=config.colors['fidelity'], 
                     alpha=0.1)
    ax3.set_xlabel('Number of Qubits', fontsize=config.label_fontsize)
    ax3.set_ylabel('Fidelity', fontsize=config.label_fontsize)
    ax3.set_title('LRET Fidelity vs Cirq FDM', fontsize=config.title_fontsize)
    ax3.set_ylim(0.0, 1.05)
    ax3.legend(fontsize=config.legend_fontsize, loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(qubits)
    
    # Annotate min fidelity
    min_idx = fidelities.index(min(fidelities))
    ax3.annotate(
        f'{fidelities[min_idx]:.4f}',
        xy=(qubits[min_idx], fidelities[min_idx]),
        xytext=(5, -15),
        textcoords='offset points',
        fontsize=config.legend_fontsize,
        color='red' if fidelities[min_idx] < 0.99 else 'green',
    )
    
    # =========================================================================
    # Plot 4: Rank Growth vs Qubits (Bottom-Right)
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.plot(qubits, ranks, '^-', 
             color=config.colors['rank'], 
             linewidth=2, 
             markersize=8,
             label='LRET Final Rank')
    
    # Add exponential reference line
    exp_ref = [2 ** (q // 2) for q in qubits]
    ax4.plot(qubits, exp_ref, 'k--', 
             linewidth=1, 
             alpha=0.5,
             label='$2^{n/2}$ Reference')
    
    ax4.set_xlabel('Number of Qubits', fontsize=config.label_fontsize)
    ax4.set_ylabel('Final Rank', fontsize=config.label_fontsize)
    ax4.set_title('Low-Rank Growth (LRET Advantage)', fontsize=config.title_fontsize)
    ax4.legend(fontsize=config.legend_fontsize)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(qubits)
    
    # Use log scale for rank if it varies greatly
    if max(ranks) > 100:
        ax4.set_yscale('log')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('./benchmarks') / f'lret_cirq_comparison_{timestamp}.{config.save_format}'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    
    logger.info(f"Benchmark plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def plot_speedup_trend(
    benchmark_result: BenchmarkResult,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
) -> Path:
    """Generate focused speedup trend plot.
    
    Args:
        benchmark_result: BenchmarkResult with metrics
        output_path: Path to save plot
        config: PlotConfig for styling
        
    Returns:
        Path to saved plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")
    
    config = config or PlotConfig()
    
    metrics = benchmark_result.metrics
    qubits = [m.qubit_count for m in metrics]
    speedups = [m.speedup_factor for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.dpi)
    
    # Main line
    ax.plot(qubits, speedups, 'o-', 
            color=config.colors['speedup'],
            linewidth=2.5,
            markersize=10,
            markerfacecolor='white',
            markeredgewidth=2)
    
    # Fill area above 1x
    ax.fill_between(qubits, 1, speedups,
                    where=[s > 1 for s in speedups],
                    color=config.colors['speedup'],
                    alpha=0.2,
                    label='LRET Advantage')
    
    # Baseline
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(qubits[-1], 1.02, 'Equal Performance', 
            ha='right', va='bottom', fontsize=9, color='gray')
    
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Speedup (LRET / Cirq FDM)', fontsize=12)
    ax.set_title('LRET Performance Advantage Over Cirq FDM', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(qubits)
    
    # Add annotations
    for i, (q, s) in enumerate(zip(qubits, speedups)):
        ax.annotate(f'{s:.1f}x', 
                   xy=(q, s),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('./benchmarks') / f'speedup_trend_{timestamp}.{config.save_format}'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Speedup plot saved to {output_path}")
    return output_path


def plot_fidelity_analysis(
    benchmark_result: BenchmarkResult,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
) -> Path:
    """Generate detailed fidelity analysis plot.
    
    Args:
        benchmark_result: BenchmarkResult with metrics
        output_path: Path to save plot
        config: PlotConfig for styling
        
    Returns:
        Path to saved plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")
    
    config = config or PlotConfig()
    
    metrics = benchmark_result.metrics
    qubits = [m.qubit_count for m in metrics]
    fidelities = [m.fidelity for m in metrics]
    trace_distances = [m.trace_distance for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=config.dpi)
    
    # Fidelity plot
    ax1.plot(qubits, fidelities, 's-', 
             color=config.colors['fidelity'],
             linewidth=2,
             markersize=8)
    ax1.axhline(y=0.99, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='99% Threshold')
    ax1.axhline(y=0.999, color='blue', linestyle=':', linewidth=1.5, 
                alpha=0.7, label='99.9% Threshold')
    ax1.set_xlabel('Number of Qubits', fontsize=11)
    ax1.set_ylabel('Fidelity', fontsize=11)
    ax1.set_title('State Fidelity', fontsize=12, fontweight='bold')
    ax1.set_ylim(min(0.95, min(fidelities) - 0.01), 1.005)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(qubits)
    
    # Trace distance plot
    ax2.semilogy(qubits, trace_distances, 'd-', 
                 color=config.colors['speedup'],
                 linewidth=2,
                 markersize=8)
    ax2.axhline(y=0.01, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='1% Threshold')
    ax2.set_xlabel('Number of Qubits', fontsize=11)
    ax2.set_ylabel('Trace Distance', fontsize=11)
    ax2.set_title('Trace Distance (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(qubits)
    
    plt.suptitle('LRET Accuracy Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('./benchmarks') / f'fidelity_analysis_{timestamp}.{config.save_format}'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Fidelity plot saved to {output_path}")
    return output_path


def generate_benchmark_report(
    benchmark_result: BenchmarkResult,
    output_path: Optional[Union[str, Path]] = None,
    include_plots: bool = True,
    plot_config: Optional[PlotConfig] = None,
) -> Path:
    """Generate comprehensive Markdown report with embedded plots.
    
    Args:
        benchmark_result: BenchmarkResult with metrics
        output_path: Path for output report
        include_plots: Whether to generate and reference plots
        plot_config: Configuration for plots
        
    Returns:
        Path to generated report
    """
    config = plot_config or PlotConfig()
    
    # Generate output path
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('./benchmarks') / f'benchmark_report_{timestamp}.md'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics = benchmark_result.metrics
    summary = benchmark_result.summary or benchmark_result.compute_summary()
    
    # Generate plots if requested
    plot_paths = {}
    if include_plots and metrics:
        try:
            comparison_plot = plot_lret_cirq_comparison(
                benchmark_result,
                output_path.parent / f'comparison_plot.{config.save_format}',
                config,
            )
            plot_paths['comparison'] = comparison_plot.name
            
            speedup_plot = plot_speedup_trend(
                benchmark_result,
                output_path.parent / f'speedup_trend.{config.save_format}',
                config,
            )
            plot_paths['speedup'] = speedup_plot.name
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    # Build report content
    report_lines = [
        "# LRET vs Cirq FDM Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    if summary:
        report_lines.extend([
            f"- **Total Benchmark Runs:** {summary.get('total_runs', len(metrics))}",
            f"- **Qubit Range:** {summary.get('qubit_range', (0, 0))[0]} - {summary.get('qubit_range', (0, 0))[1]}",
            f"- **Average Speedup:** {summary.get('avg_speedup', 0):.2f}x",
            f"- **Maximum Speedup:** {summary.get('max_speedup', 0):.2f}x",
            f"- **Average Fidelity:** {summary.get('avg_fidelity', 0):.4f}",
            f"- **Minimum Fidelity:** {summary.get('min_fidelity', 0):.4f}",
            f"- **Average Rank:** {summary.get('avg_rank', 0):.1f}",
            "",
        ])
    
    # Performance section
    report_lines.extend([
        "## Performance Analysis",
        "",
        "### Speedup Across Qubit Counts",
        "",
    ])
    
    if 'speedup' in plot_paths:
        report_lines.append(f"![Speedup Trend]({plot_paths['speedup']})")
        report_lines.append("")
    
    # Results table
    report_lines.extend([
        "### Detailed Results",
        "",
        "| Qubits | Depth | LRET Time (ms) | Cirq FDM Time (ms) | Speedup | Fidelity | Rank |",
        "|--------|-------|----------------|---------------------|---------|----------|------|",
    ])
    
    for m in metrics:
        report_lines.append(
            f"| {m.qubit_count} | {m.circuit_depth} | "
            f"{m.lret_time_ms:.2f} | {m.cirq_fdm_time_ms:.2f} | "
            f"{m.speedup_factor:.2f}x | {m.fidelity:.4f} | {m.lret_final_rank} |"
        )
    
    report_lines.append("")
    
    # Comparison plot
    if 'comparison' in plot_paths:
        report_lines.extend([
            "### Comparison Visualization",
            "",
            f"![Comparison Plot]({plot_paths['comparison']})",
            "",
        ])
    
    # Key observations
    report_lines.extend([
        "## Key Observations",
        "",
    ])
    
    if summary:
        avg_speedup = summary.get('avg_speedup', 0)
        avg_fidelity = summary.get('avg_fidelity', 0)
        max_rank = summary.get('max_rank', 0)
        
        if avg_speedup > 2:
            report_lines.append(f"- **Significant speedup** achieved with {avg_speedup:.1f}x average improvement over Cirq FDM")
        elif avg_speedup > 1:
            report_lines.append(f"- **Moderate speedup** of {avg_speedup:.1f}x compared to Cirq FDM")
        else:
            report_lines.append("- Performance comparable to Cirq FDM for these circuit sizes")
        
        if avg_fidelity >= 0.99:
            report_lines.append(f"- **Excellent accuracy** maintained with {avg_fidelity:.4f} average fidelity")
        elif avg_fidelity >= 0.95:
            report_lines.append(f"- **Good accuracy** with {avg_fidelity:.4f} average fidelity")
        else:
            report_lines.append(f"- Fidelity may need attention: {avg_fidelity:.4f} average")
        
        report_lines.append(f"- Low-rank representation stayed efficient with max rank of {max_rank}")
        report_lines.append("")
    
    # Methodology
    report_lines.extend([
        "## Methodology",
        "",
        "This benchmark compares LRET (Low-Rank Evolution Truncation) with Cirq's",
        "Full Density Matrix (FDM) simulator across varying qubit counts.",
        "",
        "- **LRET**: Uses low-rank approximation for efficient simulation",
        "- **Cirq FDM**: Full density matrix simulation for reference",
        "- **Fidelity**: Classical fidelity computed from measurement distributions",
        "- **Speedup**: Ratio of Cirq FDM time to LRET time",
        "",
        "---",
        "",
        f"*Report generated by Proxima Agent - LRET Cirq Scalability Integration*",
    ])
    
    # Write report
    report_content = '\n'.join(report_lines)
    output_path.write_text(report_content, encoding='utf-8')
    
    logger.info(f"Benchmark report generated: {output_path}")
    return output_path


def create_summary_table(benchmark_result: BenchmarkResult) -> str:
    """Create ASCII table summary for TUI display.
    
    Args:
        benchmark_result: BenchmarkResult with metrics
        
    Returns:
        Formatted ASCII table string
    """
    metrics = benchmark_result.metrics
    if not metrics:
        return "No benchmark data available."
    
    # Header
    lines = [
        "┌─────────┬───────┬────────────┬──────────────┬─────────┬──────────┬──────┐",
        "│ Qubits  │ Depth │ LRET (ms)  │ Cirq (ms)    │ Speedup │ Fidelity │ Rank │",
        "├─────────┼───────┼────────────┼──────────────┼─────────┼──────────┼──────┤",
    ]
    
    for m in metrics:
        lines.append(
            f"│ {m.qubit_count:^7} │ {m.circuit_depth:^5} │ "
            f"{m.lret_time_ms:^10.2f} │ {m.cirq_fdm_time_ms:^12.2f} │ "
            f"{m.speedup_factor:^7.2f} │ {m.fidelity:^8.4f} │ {m.lret_final_rank:^4} │"
        )
    
    lines.append("└─────────┴───────┴────────────┴──────────────┴─────────┴──────────┴──────┘")
    
    # Summary
    summary = benchmark_result.summary or benchmark_result.compute_summary()
    if summary:
        lines.extend([
            "",
            f"Average Speedup: {summary.get('avg_speedup', 0):.2f}x",
            f"Average Fidelity: {summary.get('avg_fidelity', 0):.4f}",
            f"Qubit Range: {summary.get('qubit_range', (0, 0))[0]}-{summary.get('qubit_range', (0, 0))[1]}",
        ])
    
    return '\n'.join(lines)
