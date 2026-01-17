# Benchmarking

Learn how to run, compare, and interpret Proxima benchmarks.

---

## Introduction to Benchmarking

### What Benchmarking Measures

Proxima benchmarking captures comprehensive performance metrics:

| Metric | Description |
|--------|-------------|
| Execution time (ms) | Wall-clock time for circuit execution |
| Throughput (shots/s) | Number of measurement shots per second |
| Memory peak (MB) | Maximum memory consumption during execution |
| Memory baseline (MB) | Memory usage before execution started |
| CPU usage (%) | Average CPU utilization during execution |
| GPU usage (%) | GPU utilization when using GPU backends |
| Success rate (%) | Percentage of runs that completed successfully |

### Why Benchmark Quantum Simulations

- **Backend selection:** Identify the fastest backend for your circuit size and structure
- **Regression detection:** Catch performance degradations after updates
- **Resource planning:** Validate memory and compute requirements before long jobs
- **Cost optimization:** Choose the most cost-effective backend for your workload

### When to Use Benchmarking

- Before running large production simulations
- When switching hardware (e.g., adding GPU)
- After upgrading backend libraries
- When profiling new or modified circuits
- For continuous performance monitoring

---

## Running Benchmarks

### Command Examples for Single Benchmarks
```bash
proxima benchmark run circuits/bell.json --backend lret --shots 1024 --runs 5 --warmup 1
```
- Outputs average/min/max/median/stddev times and success rate.
- Save results with `--output results/bell-lret.json`.

### Backend comparison
```bash
proxima benchmark compare circuits/bell.json --backends lret,cirq,qsim --shots 1024 --runs 3
```
- Ranks backends, shows speedup factors, and stores results in the registry.

### Interpreting results
- **Execution time (ms):** lower is faster; use median for stability.
- **Throughput:** shots per second; use when shots dominate cost.
- **Memory peak (MB):** watch for growth across runs; high variance can indicate leaks.
- **CPU/GPU usage:** helps decide between CPU vs GPU backends.
- **Success rate (%):** <100% signals backend or circuit issues.

## Benchmark History
- **View history:** `proxima benchmark history --limit 20 --backend qsim`
- **Filter/search:** `proxima benchmark list --backend cuquantum --min-time 50 --max-time 500`
- **Export:** `proxima benchmark export --format csv --output exports/benchmarks.csv`

## Advanced Features
- **Custom suites:** place YAML in `configs/benchmark_suites/quick.yaml` and run
  ```bash
  proxima benchmark suite quick --output out/quick.json
  ```
- **Scheduled benchmarking:**
  ```bash
  proxima benchmark schedule start
  proxima benchmark schedule add quick --cron "0 3 * * *"  # daily at 3 AM
  proxima benchmark schedule status
  ```
- **Performance profiling:** `proxima benchmark profile --backend cuquantum --shots 4096 --runs 5`
- **Reports:** `proxima benchmark report --format markdown --output reports/benchmarks.md`
