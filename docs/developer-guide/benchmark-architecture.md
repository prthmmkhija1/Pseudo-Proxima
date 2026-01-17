# Benchmark Architecture

This document explains how benchmarking components work together inside Proxima.

## Component Diagram (textual)
- CLI (`proxima benchmark ...`)
- BenchmarkRunner (executes benchmarks)
- BackendRegistry (resolves backends)
- ResourceMonitor (CPU/memory/GPU sampling)
- BenchmarkTimer (high-res timer)
- BenchmarkRegistry (SQLite storage)
- StatisticsCalculator (aggregations/trends/outliers)
- BackendComparator (multi-backend speedup)
- BenchmarkSuite (multi-circuit automation)
- BenchmarkScheduler (cron scheduling)
- VisualizationDataBuilder (tables/graphs for reports)

## BenchmarkRunner Workflow
1) Resolve backend via BackendRegistry
2) Start ResourceMonitor and BenchmarkTimer
3) Execute backend.run/execute
4) Stop timer/monitor; collect metrics (time, memory, CPU, GPU, throughput, success)
5) Aggregate per-run stats (avg/min/max/median/stddev, success rate)
6) Persist to BenchmarkRegistry when enabled
7) Return BenchmarkResult with metrics + metadata

## Registry Storage Format
- SQLite table `benchmarks` with columns: id, circuit_hash, backend_name, timestamp, status, execution_time_ms, memory_peak_mb, memory_baseline_mb, throughput_shots_per_sec, success_rate_percent, cpu_usage_percent, gpu_usage_percent, metadata_json, circuit_info_json, qubit_count, error_message.
- Indices on backend_name, circuit_hash, timestamp for fast filtering.

## Integration Points with Execution Pipeline
- `Executor.enable_benchmarking(runner, runs)` toggles benchmark mode.
- `Executor._maybe_run_benchmark(plan)` extracts circuit/backend from plan and runs BenchmarkRunner.
- `proxima run --benchmark` wires benchmarking into standard runs and displays a summary.

## Scheduling Pipeline
- `BenchmarkScheduler` (APScheduler) wraps cron-based jobs.
- CLI: `proxima benchmark schedule start|stop|status|add --cron "0 2 * * *"`.
- Job function loads suite YAML, creates BenchmarkRunner/Registry, executes suite, saves results.
