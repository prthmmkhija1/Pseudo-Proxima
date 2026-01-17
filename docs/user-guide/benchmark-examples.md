# Benchmark Examples

Concrete walk-throughs with expected outputs and screenshots.

---

## Example: Benchmarking a Simple Circuit

```bash
proxima benchmark run circuits/bell.json --backend lret --shots 512 --runs 3
```

**Expected Output:**
```
┌─────────────────────────────────────┐
│        Benchmark: lret              │
├─────────────────┬───────────────────┤
│ Metric          │             Value │
├─────────────────┼───────────────────┤
│ Average time (ms) │            12.45 │
│ Min time (ms)     │            11.23 │
│ Max time (ms)     │            14.67 │
│ Median (ms)       │            12.01 │
│ Stddev (ms)       │             1.42 │
│ Success rate (%)  │           100.00 │
└─────────────────┴───────────────────┘
```

---

## Example: Comparing All Backends

```bash
proxima benchmark compare circuits/qft_4.json \
  --backends lret,cirq,qiskit,quest,qsim,cuquantum \
  --shots 1024 --runs 3
```

**Expected Output:**
```
┌──────────────────────────────────────────────────────────────┐
│                    Backend Comparison                         │
├────────────┬──────────┬─────────┬──────────┬─────────────────┤
│ Backend    │ Avg (ms) │ Min     │ Max      │ Speedup Factor  │
├────────────┼──────────┼─────────┼──────────┼─────────────────┤
│ cuquantum  │    15.23 │   14.01 │    17.45 │           1.00x │
│ qsim       │    28.67 │   26.12 │    31.22 │           1.88x │
│ quest      │    45.89 │   42.34 │    49.44 │           3.01x │
│ cirq       │    67.12 │   63.45 │    70.79 │           4.41x │
│ qiskit     │    89.34 │   85.67 │    93.01 │           5.87x │
│ lret       │   112.45 │  108.23 │   116.67 │           7.38x │
└────────────┴──────────┴─────────┴──────────┴─────────────────┘

✅ Winner: cuquantum
```

---

## Example: Analyzing Performance Trends

```bash
proxima benchmark stats --backend qsim --last 20
```

**Expected Output:**
```
┌────────────────────────────────────────────┐
│      Statistics: qsim (last 20 runs)       │
├────────────────────┬───────────────────────┤
│ Metric             │                 Value │
├────────────────────┼───────────────────────┤
│ Mean time (ms)     │                 28.45 │
│ Median time (ms)   │                 27.89 │
│ Std deviation      │                  2.34 │
│ P25                │                 26.12 │
│ P75                │                 30.01 │
│ P95                │                 33.45 │
├────────────────────┼───────────────────────┤
│ Trend              │            increasing │
│ Slope (ms/run)     │                 +0.12 │
│ Outliers detected  │                     1 │
│ Success rate       │                 100%  │
└────────────────────┴───────────────────────┘
```

---

## Example: Creating a Custom Benchmark Suite

**Step 1:** Create `configs/benchmark_suites/custom.yaml`:

```yaml
name: custom
circuits:
  - bell_state
  - ghz_3
  - qft_4
backends:
  - lret
  - qsim
shots: 1024
runs: 3
```

**Step 2:** Run the suite:

```bash
proxima benchmark suite custom --output out/custom.json
```

**Expected Output:**
```
Running benchmark suite: custom
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1/6] bell_state on lret...    ✓  12.3 ms
[2/6] bell_state on qsim...    ✓   5.1 ms
[3/6] ghz_3 on lret...         ✓  18.7 ms
[4/6] ghz_3 on qsim...         ✓   7.2 ms
[5/6] qft_4 on lret...         ✓  45.6 ms
[6/6] qft_4 on qsim...         ✓  15.8 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Suite Summary:
  Total benchmarks: 6
  Avg time: 17.45 ms
  Min time: 5.1 ms
  Max time: 45.6 ms

Results saved to: out/custom.json
```
