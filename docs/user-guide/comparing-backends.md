# Comparing Backends

## Plan comparisons
```bash
proxima compare circuits/bell.json --backends local,qiskit,ibm
```
- Planner selects sequential or parallel strategy
- Uses timeouts and resource checks from config

## Metrics
- Execution time per backend
- Memory peak estimates
- Result agreement and pairwise fidelity
- Recommended backend with reason

## Reports
- Summary in CLI
- Detailed export via `proxima export --format json|csv|html`

## Tips
- Keep identical shot counts across backends
- Use `--timeout-ms` to cap slow backends
- Review warnings for partial or failed runs
