# Running Simulations

## Local execution
```bash
proxima run path/to/circuit.json --backend local
```
- Uses built-in simulator
- Fast feedback, no network

## Remote execution
```bash
proxima run path/to/circuit.json --backend qiskit --shots 1024
```
- Ensure credentials are configured for the backend

## Execution options
- `--shots`: number of shots
- `--seed`: random seed
- `--timeout-ms`: per-backend timeout
- `--dry-run`: validate only

## Results
- Inspect counts, duration, metadata
- Export via `proxima export`
