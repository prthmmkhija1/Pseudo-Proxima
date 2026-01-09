# Testing

## Strategy
- Test pyramid: Unit (60%), Integration (30%), E2E (10%)
- Markers: `unit`, `integration`, `e2e`, `backend`, `performance`, `slow`, `requires_network`

## Running tests
```bash
pytest -m "unit"
pytest -m "integration"
pytest -m "e2e"
pytest -m "performance" --run-slow
```

## Coverage
```bash
pytest --cov=proxima --cov-report=html
```

## Benchmarks
- Uses `pytest-benchmark`
- Slow benchmarks are skipped unless `--run-slow` is set

## Tips
- Use fixtures from `tests/conftest.py`
- Keep network-dependent tests behind `--run-network`
- Prefer mock adapters for backend tests
