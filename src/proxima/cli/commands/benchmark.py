"""Benchmark CLI commands (Phase 6).

Implements benchmark run/compare/history/stats/profile/cleanup/export/report.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, List

import yaml

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from proxima.backends.registry import BackendRegistry
from proxima.benchmarks.comparator import BackendComparator
from proxima.benchmarks.profiler import BackendProfiler
from proxima.benchmarks.runner import BenchmarkRunner
from proxima.benchmarks.statistics import StatisticsCalculator
from proxima.benchmarks.suite import BenchmarkSuite
from proxima.benchmarks.visualization import VisualizationDataBuilder
from proxima.benchmarks.scheduler import BenchmarkScheduler
from proxima.data.benchmark_registry import BenchmarkRegistry as ResultRegistry
from proxima.data.metrics import BenchmarkResult

console = Console()
app = typer.Typer(help="Benchmark commands")
schedule_app = typer.Typer(help="Benchmark scheduling")
_scheduler = BenchmarkScheduler()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_circuit(circuit: str | Path) -> Any:
    path = Path(circuit)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return circuit


def _get_runner_and_registry() -> tuple[BenchmarkRunner, ResultRegistry, BackendRegistry]:
    backend_registry = BackendRegistry()
    backend_registry.discover()
    result_registry = ResultRegistry()
    runner = BenchmarkRunner(backend_registry, results_storage=result_registry)
    return runner, result_registry, backend_registry


def _results_to_dataframe(results: Iterable[BenchmarkResult]):
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required for this operation") from exc

    rows: list[dict[str, Any]] = []
    for r in results:
        if not r.metrics:
            continue
        rows.append(
            {
                "id": r.benchmark_id,
                "backend": r.metrics.backend_name,
                "timestamp": r.metrics.timestamp,
                "time_ms": r.metrics.execution_time_ms,
                "status": getattr(r.status, "value", str(r.status)),
                "success_rate": r.metrics.success_rate_percent,
                "memory_peak_mb": r.metrics.memory_peak_mb,
                "qubits": (r.metrics.circuit_info or {}).get("qubit_count"),
                "depth": (r.metrics.circuit_info or {}).get("depth"),
                "error": r.error_message,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# benchmark run
# ---------------------------------------------------------------------------
@app.command("run")
def benchmark_run(
    circuit: str = typer.Argument(..., help="Circuit string or path to benchmark"),
    backend: str = typer.Option(..., "--backend", "-b", help="Backend to use"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of shots"),
    runs: int = typer.Option(5, "--runs", "-n", help="Number of benchmark runs"),
    warmup: int = typer.Option(2, "--warmup", "-w", help="Warmup runs"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Path to save results"),
) -> None:
    """Run benchmark suite on a single backend."""

    runner, registry, _ = _get_runner_and_registry()
    circuit_obj = _load_circuit(circuit)
    result = runner.run_benchmark_suite(
        circuit=circuit_obj,
        backend_name=backend,
        num_runs=runs,
        shots=shots,
        warmup_runs=warmup,
    )

    stats = result.metadata.get("statistics", {}) if result.metadata else {}
    table = Table(title=f"Benchmark: {backend}", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Average time (ms)", f"{stats.get('avg_time_ms', 0):.2f}")
    table.add_row("Min time (ms)", f"{stats.get('min_time_ms', 0):.2f}")
    table.add_row("Max time (ms)", f"{stats.get('max_time_ms', 0):.2f}")
    table.add_row("Median (ms)", f"{stats.get('median_time_ms', 0):.2f}")
    table.add_row("Stddev (ms)", f"{stats.get('stddev_time_ms', 0):.2f}")
    table.add_row("Success rate (%)", f"{stats.get('success_rate_percent', 0):.2f}")
    console.print(table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        console.print(f"Saved benchmark result to {output}")


# ---------------------------------------------------------------------------
# benchmark compare
# ---------------------------------------------------------------------------
@app.command("compare")
def benchmark_compare(
    circuit: str = typer.Argument(..., help="Circuit string or path"),
    backends: List[str] | None = typer.Option(None, "--backends", "-b", help="Backends to compare"),
    all: bool = typer.Option(False, "--all", help="Use all available backends"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of shots"),
    runs: int = typer.Option(5, "--runs", "-n", help="Runs per backend"),
) -> None:
    """Compare performance across backends."""

    runner, registry, backend_registry = _get_runner_and_registry()
    circuit_obj = _load_circuit(circuit)

    if all:
        backend_names = backend_registry.list_available()
    else:
        if not backends:
            raise typer.BadParameter("Provide --backends or use --all")
        backend_names = backends

    comparator = BackendComparator(runner, backend_registry, results_registry=registry)
    comparison = comparator.compare_backends(
        circuit=circuit_obj,
        backend_names=backend_names,
        shots=shots,
        num_runs=runs,
    )

    table = Table(title="Backend Comparison", box=box.SIMPLE_HEAVY)
    table.add_column("Backend", style="cyan")
    table.add_column("Avg time (ms)", justify="right")
    table.add_column("Speedup", justify="right")

    fastest = comparison.winner
    for res in sorted(comparison.results, key=lambda r: r.metrics.execution_time_ms if r.metrics else float("inf")):
        if not res.metrics:
            continue
        speed = comparison.speedup_factors.get(res.metrics.backend_name, 0.0)
        style = "green" if res.metrics.backend_name == fastest else None
        table.add_row(res.metrics.backend_name, f"{res.metrics.execution_time_ms:.2f}", f"{speed:.2f}", style=style)

    console.print(table)


# ---------------------------------------------------------------------------
# benchmark list
# ---------------------------------------------------------------------------
@app.command("list")
def benchmark_list(
    backend: str | None = typer.Option(None, "--backend", "-b", help="Filter by backend"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table|json|csv"),
) -> None:
    """List stored benchmark results."""

    _, registry, _ = _get_runner_and_registry()
    filters = {"backend_name": backend} if backend else None
    results = registry.get_results_filtered(filters=filters, limit=limit)

    if format == "json":
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        return

    if format == "csv":
        df = _results_to_dataframe(results)
        typer.echo(df.to_csv(index=False))
        return

    table = Table(title="Benchmarks", box=box.SIMPLE_HEAVY)
    table.add_column("ID", overflow="fold")
    table.add_column("Backend")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Status")
    table.add_column("Timestamp")
    for r in results:
        ts = r.metrics.timestamp.isoformat() if r.metrics else ""
        time_ms = f"{r.metrics.execution_time_ms:.2f}" if r.metrics else ""
        status = getattr(r.status, "value", str(r.status))
        table.add_row(r.benchmark_id, r.metrics.backend_name if r.metrics else "", time_ms, status, ts)
    console.print(table)


# ---------------------------------------------------------------------------
# benchmark history
# ---------------------------------------------------------------------------
@app.command("history")
def benchmark_history(
    backend: str | None = typer.Option(None, "--backend", "-b", help="Filter by backend"),
    circuit: str | None = typer.Option(None, "--circuit", "-c", help="Filter by circuit hash"),
    days: int = typer.Option(7, "--days", "-d", help="Look back N days"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
) -> None:
    """Show recent benchmark history."""

    _, registry, _ = _get_runner_and_registry()
    end = datetime.now()
    start = end - timedelta(days=days)
    results = registry.get_results_in_range(start, end, limit=limit)
    if backend:
        results = [r for r in results if r.metrics and r.metrics.backend_name == backend]
    if circuit:
        results = [r for r in results if r.circuit_hash == circuit]

    table = Table(title="History", box=box.SIMPLE_HEAVY)
    table.add_column("Timestamp")
    table.add_column("Backend")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Status")
    for r in results:
        ts = r.metrics.timestamp.isoformat() if r.metrics else ""
        time_ms = f"{r.metrics.execution_time_ms:.2f}" if r.metrics else ""
        status = getattr(r.status, "value", str(r.status))
        table.add_row(ts, r.metrics.backend_name if r.metrics else "", time_ms, status)
    console.print(table)


# ---------------------------------------------------------------------------
# benchmark stats
# ---------------------------------------------------------------------------
@app.command("stats")
def benchmark_stats(
    backend: str = typer.Argument(..., help="Backend to analyze"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed stats"),
) -> None:
    """Compute statistics for a backend."""

    _, registry, _ = _get_runner_and_registry()
    results = registry.get_results_for_backend(backend, limit=None)
    if not results:
        console.print(f"No results found for backend '{backend}'")
        return

    stats_calc = StatisticsCalculator()
    profiler = BackendProfiler(registry)
    exec_times = [r.metrics.execution_time_ms for r in results if r.metrics]
    basic = stats_calc.calculate_basic_stats(exec_times)
    trend = stats_calc.analyze_trends(results)
    outliers = stats_calc.detect_outliers(results)

    # Calculate success rate
    success_count = sum(1 for r in results if r.metrics and r.metrics.success_rate_percent == 100.0)
    success_rate = (success_count / len(results) * 100.0) if results else 0.0

    lines = [
        f"Mean: {basic['mean']:.2f} ms",
        f"Median: {basic['median']:.2f} ms",
        f"Min/Max: {basic['min']:.2f} / {basic['max']:.2f} ms",
        f"Stddev: {basic['stdev']:.2f} ms",
        f"Success rate: {success_rate:.1f}%",
        f"Trend: {trend.direction} (slope {trend.slope:.6f})",
        f"Outliers: {len(outliers)}",
    ]

    if detailed:
        pct = stats_calc.calculate_percentiles(exec_times, [50, 95, 99])
        lines.append(f"P50/P95/P99: {pct[50]:.2f} / {pct[95]:.2f} / {pct[99]:.2f} ms")
        # Add recommendations
        profile = profiler.generate_profile(backend)
        if profile.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in profile.recommendations:
                lines.append(f"  • {rec}")
    panel = Panel("\n".join(lines), title=f"Stats: {backend}", box=box.SIMPLE_HEAVY)
    console.print(panel)


# ---------------------------------------------------------------------------
# benchmark profile
# ---------------------------------------------------------------------------
@app.command("profile")
def benchmark_profile(
    backend: str = typer.Argument(..., help="Backend to profile"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save profile to file"),
) -> None:
    """Generate backend performance profile."""

    _, registry, _ = _get_runner_and_registry()
    profiler = BackendProfiler(registry)
    profile = profiler.generate_profile(backend)
    console.print(profile.to_report())

    if output:
        payload = {
            "backend_name": profile.backend_name,
            "total_benchmarks": profile.total_benchmarks,
            "average_execution_time_ms": profile.average_execution_time_ms,
            "performance_by_qubit_count": profile.performance_by_qubit_count,
            "performance_by_depth": profile.performance_by_depth,
            "memory_usage_trend": profile.memory_usage_trend,
            "optimal_circuit_sizes": profile.optimal_circuit_sizes,
            "recommendations": profile.recommendations,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        console.print(f"Saved profile to {output}")


# ---------------------------------------------------------------------------
# benchmark suite
# ---------------------------------------------------------------------------
@app.command("suite")
def benchmark_suite(
    suite_name: str = typer.Argument(..., help="Predefined or custom suite name"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Directory to save results"),
) -> None:
    """Execute a benchmark suite defined in YAML."""

    suite_path = Path("configs/benchmark_suites") / f"{suite_name}.yaml"
    if not suite_path.exists():
        raise typer.BadParameter(f"Suite definition not found: {suite_path}")

    data = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    suite = BenchmarkSuite(
        name=data.get("name", suite_name),
        circuits=data.get("circuits", []),
        backends=data.get("backends", []),
        shots=int(data.get("shots", 1024)),
        runs=int(data.get("runs", 3)),
    )

    backend_registry = BackendRegistry()
    backend_registry.discover()
    registry = ResultRegistry()
    runner = BenchmarkRunner(backend_registry, results_storage=registry)

    with Progress() as progress:
        task = progress.add_task(f"Running suite: {suite.name}", total=len(suite.circuits) * len(suite.backends))

        def _cb(done: int, total: int, backend: str, circuit: Any) -> None:
            progress.update(task, completed=done, total=total, description=f"{backend} :: {str(circuit)[:20]}")

        results = suite.execute(runner, registry=registry, progress_callback=_cb)

    console.print(f"Suite '{suite.name}' complete. Avg time: {results.summary.get('avg_time_ms', 0):.2f} ms")

    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(results.summary, indent=2, default=str))
        (output / "results.json").write_text(json.dumps(results.to_dict(), indent=2, default=str))
        console.print(f"Saved suite results to {output}")


# ---------------------------------------------------------------------------
# benchmark cleanup
# ---------------------------------------------------------------------------
@app.command("cleanup")
def benchmark_cleanup(
    days: int = typer.Option(30, "--days", "-d", help="Delete results older than N days"),
    backup: bool = typer.Option(False, "--backup", help="Create backup before cleanup"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt"),
) -> None:
    """Cleanup old benchmark results."""

    _, registry, _ = _get_runner_and_registry()

    if backup:
        backup_path = registry.create_backup()
        console.print(f"Backup created at {backup_path}")

    if not confirm:
        typer.confirm(f"Delete results older than {days} days?", abort=True)

    deleted = registry.delete_results_older_than(days)
    console.print(f"Deleted {deleted} results")
    saved_bytes = registry.vacuum_database()
    console.print(f"Vacuum saved {saved_bytes} bytes")


# ---------------------------------------------------------------------------
# benchmark export
# ---------------------------------------------------------------------------
@app.command("export")
def benchmark_export(
    output_path: Path = typer.Argument(..., help="Where to save export"),
    format: str = typer.Option("json", "--format", "-f", help="json|csv|excel"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="Filter by backend"),
    days: int | None = typer.Option(None, "--days", "-d", help="Export last N days"),
) -> None:
    """Export benchmark data."""

    _, registry, _ = _get_runner_and_registry()

    results: list[BenchmarkResult]
    if days is not None:
        end = datetime.now()
        start = end - timedelta(days=days)
        results = registry.get_results_in_range(start, end, limit=None)
    else:
        filters = {"backend_name": backend} if backend else None
        results = registry.get_results_filtered(filters=filters, limit=None)

    if backend:
        results = [r for r in results if r.metrics and r.metrics.backend_name == backend]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        output_path.write_text(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        console.print(f"Exported {len(results)} results to {output_path}")
        return

    df = _results_to_dataframe(results)
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "excel":
        df.to_excel(output_path, index=False)
    else:
        raise typer.BadParameter("Unsupported format. Use json|csv|excel")
    console.print(f"Exported {len(df)} results to {output_path}")


# ---------------------------------------------------------------------------
# benchmark report
# ---------------------------------------------------------------------------
@app.command("report")
def benchmark_report(
    backend: str | None = typer.Option(None, "--backend", "-b", help="Specific backend or all"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save report to file"),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown|html|pdf"),
) -> None:
    """Generate benchmark report."""

    _, registry, _ = _get_runner_and_registry()
    stats_calc = StatisticsCalculator()
    profiler = BackendProfiler(registry)
    viz = VisualizationDataBuilder(registry)

    if backend:
        results = registry.get_results_for_backend(backend, limit=None)
        profile = profiler.generate_profile(backend)
        exec_times = [r.metrics.execution_time_ms for r in results if r.metrics]
        basic = stats_calc.calculate_basic_stats(exec_times)
        trend = stats_calc.analyze_trends(results)
        recs = "\n".join(f"- {r}" for r in profile.recommendations) or "(none)"
        body = [
            f"# Benchmark Report: {backend}",
            "",
            "## Executive Summary",
            f"- Total benchmarks: {len(results)}",
            f"- Average execution time: {basic['mean']:.2f} ms",
            f"- Performance trend: {trend.direction}",
            "",
            "## Statistical Analysis",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean | {basic['mean']:.2f} ms |",
            f"| Median | {basic['median']:.2f} ms |",
            f"| Min | {basic['min']:.2f} ms |",
            f"| Max | {basic['max']:.2f} ms |",
            f"| Std Dev | {basic['stdev']:.2f} ms |",
            f"| Trend Slope | {trend.slope:.6f} |",
            "",
            "## Recommendations",
            recs,
        ]
    else:
        results = registry.get_results_filtered(limit=None)
        df = viz.build_time_series(results)
        unique_backends = sorted(df["backend_name"].unique()) if not df.empty else []
        # Compute per-backend stats for comparison
        comparison_rows = []
        for be in unique_backends:
            be_results = [r for r in results if r.metrics and r.metrics.backend_name == be]
            if be_results:
                times = [r.metrics.execution_time_ms for r in be_results if r.metrics]
                avg_t = sum(times) / len(times) if times else 0.0
                comparison_rows.append(f"| {be} | {len(be_results)} | {avg_t:.2f} ms |")
        comparison_table = "\n".join(comparison_rows) if comparison_rows else "(no data)"
        body = [
            "# Benchmark Report (All Backends)",
            "",
            "## Executive Summary",
            f"- Total benchmarks: {len(results)}",
            f"- Backends analyzed: {len(unique_backends)}",
            "",
            "## Backend Performance Comparison",
            "| Backend | Benchmarks | Avg Time |",
            "|---------|------------|----------|",
            comparison_table,
        ]

    report_md = "\n\n".join(body)

    if output is None:
        console.print(report_md)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    if format == "markdown":
        output.write_text(report_md, encoding="utf-8")
        console.print(f"Saved markdown report to {output}")
    elif format == "html":
        html = f"<html><body><pre>{report_md}</pre></body></html>"
        output.write_text(html, encoding="utf-8")
        console.print(f"Saved HTML report to {output}")
    elif format == "pdf":
        try:
            import weasyprint  # type: ignore

            html = f"<html><body><pre>{report_md}</pre></body></html>"
            weasyprint.HTML(string=html).write_pdf(str(output))
            console.print(f"Saved PDF report to {output}")
        except Exception:
            # Fallback: save markdown if PDF generation unavailable
            fallback = output.with_suffix(".md")
            fallback.write_text(report_md, encoding="utf-8")
            console.print(f"PDF generation not available; saved markdown to {fallback}")
    else:
        raise typer.BadParameter("Unsupported format. Use markdown|html|pdf")


# ---------------------------------------------------------------------------
# benchmark schedule (start/stop/status/add)
# ---------------------------------------------------------------------------


@schedule_app.command("start")
def schedule_start():
    """Start benchmark scheduler daemon."""
    _scheduler.start()
    console.print("Scheduler started")


@schedule_app.command("stop")
def schedule_stop():
    """Stop benchmark scheduler."""
    _scheduler.stop()
    console.print("Scheduler stopped")


@schedule_app.command("status")
def schedule_status():
    """Show scheduled jobs."""
    jobs = _scheduler.status()
    if not jobs:
        console.print("No scheduled jobs")
    else:
        console.print("Scheduled jobs:")
        for line in jobs:
            console.print(f" - {line}")


def _validate_cron_field(value: str, field_name: str, max_val: int) -> bool:
    """Validate a cron field value."""
    if value == "*":
        return True
    # Handle step values like */5
    if value.startswith("*/"):
        try:
            step = int(value[2:])
            return 1 <= step <= max_val
        except ValueError:
            return False
    # Handle ranges like 1-5
    if "-" in value:
        try:
            start, end = value.split("-", 1)
            return 0 <= int(start) <= int(end) <= max_val
        except ValueError:
            return False
    # Handle comma-separated values
    if "," in value:
        return all(_validate_cron_field(v, field_name, max_val) for v in value.split(","))
    # Plain number
    try:
        num = int(value)
        return 0 <= num <= max_val
    except ValueError:
        return False


@schedule_app.command("add")
def schedule_add(
    suite_name: str = typer.Argument(..., help="Suite name to schedule"),
    cron: str = typer.Option("0 2 * * *", "--cron", help="Cron string, default 2 AM daily"),
):
    """Add a new scheduled benchmark job.

    Examples:
        proxima benchmark schedule add quick --cron "0 2 * * *"   # Daily at 2 AM
        proxima benchmark schedule add standard --cron "0 3 * * 0" # Sundays 3 AM
    """

    def _job() -> None:
        suite_path = Path("configs/benchmark_suites") / f"{suite_name}.yaml"
        if not suite_path.exists():
            console.print(f"[red]Suite not found: {suite_path}[/red]")
            return
        data = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
        suite = BenchmarkSuite(
            name=data.get("name", suite_name),
            circuits=data.get("circuits", []),
            backends=data.get("backends", []),
            shots=int(data.get("shots", 1024)),
            runs=int(data.get("runs", 3)),
        )
        backend_registry = BackendRegistry()
        backend_registry.discover()
        registry = ResultRegistry()
        runner = BenchmarkRunner(backend_registry, results_storage=registry)
        suite.execute(runner, registry=registry)
        console.print(f"[green]Scheduled benchmark '{suite_name}' completed[/green]")

    # Parse and validate cron expression
    parts = cron.split()
    if len(parts) != 5:
        raise typer.BadParameter("Cron must have 5 fields: minute hour day month day_of_week")

    minute, hour, day, month, dow = parts
    # Validate each field
    field_limits = [("minute", minute, 59), ("hour", hour, 23), ("day", day, 31), ("month", month, 12), ("day_of_week", dow, 6)]
    for field_name, value, max_val in field_limits:
        if not _validate_cron_field(value, field_name, max_val):
            raise typer.BadParameter(f"Invalid {field_name} value: {value}")

    job_id = _scheduler.add_job(
        _job, trigger="cron", minute=minute, hour=hour, day=day, month=month, day_of_week=dow
    )
    console.print(f"[green]Added scheduled job {job_id} for suite '{suite_name}'[/green]")
    console.print(f"  Schedule: {cron} (min hour day month dow)")


# Register schedule subcommands under benchmark
app.add_typer(schedule_app, name="schedule")


__all__ = ["app"]
