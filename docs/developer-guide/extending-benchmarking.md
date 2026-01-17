# Extending Benchmarking

Guidance for contributors to extend metrics, suites, and outputs.

## Adding New Metrics
- Extend `BenchmarkMetrics` with new fields; update `to_dict/from_dict`.
- Populate fields in `BenchmarkRunner.run_benchmark` and suite aggregation.
- Add display in CLI tables (`benchmark run/compare`) if user-facing.
- Update tests under `tests/benchmarks` to cover new metrics.

## Custom Benchmark Suites
- Add YAML under `configs/benchmark_suites/*.yaml` with `name`, `circuits`, `backends`, `shots`, `runs`.
- CLI: `proxima benchmark suite <name> --output out.json`.
- For programmatic creation, use `BenchmarkSuite.execute(runner, registry, progress_callback)`.

## Extending Statistical Analysis
- Add methods to `StatisticsCalculator` for new aggregates or trend analyses.
- Pipe results into CLI stats/report commands via `benchmark stats` and `benchmark report`.
- Ensure pandas/scipy optional dependencies remain optional.

## Export Formats
- `benchmark export` currently supports markdown/html/pdf (report) and csv/json (data).
- To add a new format, extend the export/report handlers in `src/proxima/cli/commands/benchmark.py` and, if needed, VisualizationDataBuilder.

## API Docstrings
- Use Sphinx-style docstrings on public classes and methods.
- Include parameter types, return types, and short usage notes.
- Keep examples minimal and runnable.
