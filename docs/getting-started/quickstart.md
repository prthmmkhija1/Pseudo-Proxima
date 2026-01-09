# Quickstart

## 1. Initialize project
```bash
proxima init --template default
```

## 2. Configure a backend
```bash
proxima config set backends.default local
proxima config set backends.timeout_s 300
```

## 3. Run a sample circuit
```bash
proxima run examples/bell.json --backend local
```

## 4. Compare multiple backends
```bash
proxima compare examples/bell.json --backends local,qiskit
```

## 5. View results
- CLI: `proxima results list`
- Export: `proxima export --format json --output out/report.json`
