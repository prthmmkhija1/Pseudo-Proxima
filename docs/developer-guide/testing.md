# Testing

This guide covers the testing strategy, tools, and best practices for Proxima development.

## Overview

Proxima uses a comprehensive testing strategy:

```

                      Testing Pyramid                         

                        E2E Tests                             
                     (Full workflows)                         
                                                             
                                                             
                                    
                Integration Tests                          
                 (Component combos)                        
                                    
                                                             
                  
                  Unit Tests                                
         (Individual functions/classes)                     
                  

```

## Test Organization

```
tests/
 conftest.py              # Shared fixtures
 unit/                    # Unit tests
    core/
       test_circuit.py
       test_execution.py
       test_result.py
       test_control.py
    backends/
       test_lret.py
       test_cirq.py
       test_qiskit.py
       test_quest.py
       test_qsim.py
       test_cuquantum.py
    intelligence/
       test_llm_router.py
       test_insights.py
    data/
        test_session.py
        test_storage.py
 integration/             # Integration tests
    test_backend_integration.py
    test_api_integration.py
    test_cli_integration.py
 e2e/                     # End-to-end tests
    test_workflow_e2e.py
    test_agent_e2e.py
 benchmarks/              # Performance tests
    bench_backends.py
    bench_circuits.py
 fixtures/                # Test data
     circuits/
        bell.json
        ghz.json
        qft.json
     results/
         expected/
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/core/test_circuit.py

# Run specific test function
pytest tests/unit/core/test_circuit.py::test_bell_state
```

### Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only e2e tests
pytest tests/e2e/

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

### Using Markers

```bash
# Run tests by marker
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m slow
pytest -m gpu

# Skip slow tests
pytest -m "not slow"

# Run backend-specific tests
pytest -m cirq
pytest -m qiskit
```

### Parallel Execution

```bash
# Run tests in parallel
pytest -n auto

# Specify number of workers
pytest -n 4
```

## Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
    gpu: GPU-required tests
    cirq: Cirq backend tests
    qiskit: Qiskit backend tests
    quest: QuEST backend tests
    qsim: qsim backend tests
    cuquantum: cuQuantum backend tests
```

### conftest.py

```python
# tests/conftest.py
import pytest
from proxima.core.circuit import Circuit
from proxima.core.execution import ExecutionEngine
from proxima.backends.registry import BackendRegistry

@pytest.fixture
def simple_circuit():
    """Create a simple 2-qubit Bell state circuit."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit

@pytest.fixture
def ghz_circuit():
    """Create a 3-qubit GHZ state circuit."""
    circuit = Circuit(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    return circuit

@pytest.fixture
def execution_engine():
    """Create execution engine instance."""
    return ExecutionEngine()

@pytest.fixture
def backend_registry():
    """Create backend registry instance."""
    return BackendRegistry()

@pytest.fixture(params=["lret", "cirq", "qiskit"])
def available_backend(request, backend_registry):
    """Parametrized fixture for available backends."""
    backend = request.param
    if not backend_registry.is_available(backend):
        pytest.skip(f"{backend} not available")
    return backend
```

## Unit Tests

### Core Module Tests

```python
# tests/unit/core/test_circuit.py
import pytest
from proxima.core.circuit import Circuit, Gate

class TestCircuit:
    def test_create_circuit(self):
        circuit = Circuit(5)
        assert circuit.qubit_count == 5
        assert len(circuit.gates) == 0
    
    def test_add_hadamard(self):
        circuit = Circuit(2)
        circuit.h(0)
        
        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "H"
        assert circuit.gates[0].targets == [0]
    
    def test_add_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        
        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "CNOT"
        assert circuit.gates[0].control == 0
        assert circuit.gates[0].target == 1
    
    def test_bell_state_circuit(self, simple_circuit):
        assert simple_circuit.qubit_count == 2
        assert len(simple_circuit.gates) == 2
    
    def test_invalid_qubit_index(self):
        circuit = Circuit(2)
        with pytest.raises(ValueError):
            circuit.h(5)  # Invalid qubit
    
    def test_circuit_depth(self):
        circuit = Circuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.cnot(0, 2)
        
        assert circuit.depth == 2
```

### Backend Tests

```python
# tests/unit/backends/test_lret.py
import pytest
import numpy as np
from proxima.backends.lret import LRETAdapter

class TestLRETAdapter:
    @pytest.fixture
    def adapter(self):
        adapter = LRETAdapter()
        adapter.initialize()
        yield adapter
        adapter.cleanup()
    
    def test_initialize(self, adapter):
        assert adapter.name == "lret"
        assert adapter._initialized is True
    
    def test_execute_bell_state(self, adapter, simple_circuit):
        result = adapter.execute(simple_circuit, shots=1000)
        
        assert result.shot_count == 1000
        assert "00" in result.counts or "11" in result.counts
        
        # Bell state should have ~50/50 distribution
        total = sum(result.counts.values())
        for state, count in result.counts.items():
            assert 0.3 < count / total < 0.7
    
    def test_supports_simulator(self, adapter):
        assert adapter.supports_simulator("state_vector") is True
        assert adapter.supports_simulator("density_matrix") is True
        assert adapter.supports_simulator("tensor_network") is False
    
    def test_estimate_resources(self, adapter, simple_circuit):
        resources = adapter.estimate_resources(simple_circuit)
        
        assert "memory_bytes" in resources
        assert resources["memory_bytes"] == 2**2 * 16  # 2 qubits
```

### Result Normalizer Tests

```python
# tests/unit/backends/test_normalizers.py
import pytest
from proxima.backends.lret.normalizer import LRETNormalizer
from proxima.backends.cirq_backend.normalizer import CirqNormalizer

class TestLRETNormalizer:
    def test_normalize_counts(self):
        normalizer = LRETNormalizer()
        raw_result = {
            "measurements": {
                "|00": 500,
                "|11": 500
            }
        }
        
        result = normalizer.normalize(raw_result)
        
        assert "00" in result.counts
        assert "11" in result.counts
        assert result.counts["00"] == 500

class TestCirqNormalizer:
    def test_normalize_counts(self):
        normalizer = CirqNormalizer()
        raw_result = {
            "measurements": [
                [0, 0],
                [1, 1],
                [0, 0],
                [1, 1]
            ]
        }
        
        result = normalizer.normalize(raw_result)
        
        assert "00" in result.counts
        assert "11" in result.counts
```

## Integration Tests

```python
# tests/integration/test_backend_integration.py
import pytest
from proxima.core.execution import execute_circuit
from proxima.core.circuit import Circuit

class TestBackendIntegration:
    @pytest.mark.integration
    def test_execute_on_all_backends(self, simple_circuit, available_backend):
        """Test execution on all available backends."""
        result = execute_circuit(
            circuit=simple_circuit,
            backend=available_backend,
            shots=1000
        )
        
        assert result.success is True
        assert result.shot_count == 1000
        assert len(result.counts) > 0
    
    @pytest.mark.integration
    def test_backend_result_consistency(self, simple_circuit):
        """Test that different backends produce similar results."""
        results = {}
        
        for backend in ["lret", "cirq", "qiskit"]:
            try:
                result = execute_circuit(
                    circuit=simple_circuit,
                    backend=backend,
                    shots=10000,
                    seed=42
                )
                results[backend] = result
            except ImportError:
                continue
        
        if len(results) < 2:
            pytest.skip("Not enough backends available")
        
        # Check statistical similarity
        for backend1, result1 in results.items():
            for backend2, result2 in results.items():
                if backend1 != backend2:
                    # Chi-square test or similar
                    assert self._results_similar(result1, result2)
```

## End-to-End Tests

```python
# tests/e2e/test_workflow_e2e.py
import pytest
import subprocess
import json

class TestWorkflowE2E:
    @pytest.mark.e2e
    def test_cli_run_command(self, tmp_path):
        """Test full CLI workflow."""
        # Create test circuit
        circuit_file = tmp_path / "circuit.json"
        circuit_data = {
            "qubits": 2,
            "gates": [
                {"type": "H", "targets": [0]},
                {"type": "CNOT", "control": 0, "target": 1}
            ]
        }
        circuit_file.write_text(json.dumps(circuit_data))
        
        # Run CLI command
        result = subprocess.run(
            ["proxima", "run", str(circuit_file), "--backend", "lret"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Execution completed" in result.stdout
    
    @pytest.mark.e2e
    def test_compare_workflow(self, tmp_path):
        """Test backend comparison workflow."""
        circuit_file = tmp_path / "circuit.json"
        circuit_data = {"qubits": 2, "gates": [{"type": "H", "targets": [0]}]}
        circuit_file.write_text(json.dumps(circuit_data))
        
        result = subprocess.run(
            ["proxima", "compare", str(circuit_file), "--backends", "lret,cirq"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Comparison" in result.stdout
```

## Benchmarks

```python
# tests/benchmarks/bench_backends.py
import pytest
from proxima.core.execution import execute_circuit
from proxima.core.circuit import Circuit

class TestBackendBenchmarks:
    @pytest.mark.benchmark
    @pytest.mark.parametrize("qubit_count", [5, 10, 15, 20])
    def test_benchmark_lret(self, benchmark, qubit_count):
        """Benchmark LRET backend."""
        circuit = self._create_random_circuit(qubit_count)
        
        result = benchmark(
            execute_circuit,
            circuit=circuit,
            backend="lret",
            shots=1000
        )
        
        assert result.success
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("shot_count", [100, 1000, 10000])
    def test_benchmark_shot_scaling(self, benchmark, shot_count):
        """Benchmark shot count scaling."""
        circuit = Circuit(5)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        result = benchmark(
            execute_circuit,
            circuit=circuit,
            backend="lret",
            shots=shot_count
        )
        
        assert result.success
    
    def _create_random_circuit(self, qubit_count: int) -> Circuit:
        """Create a random circuit for benchmarking."""
        import random
        circuit = Circuit(qubit_count)
        
        for _ in range(qubit_count * 10):
            gate = random.choice(["H", "X", "Y", "Z"])
            qubit = random.randint(0, qubit_count - 1)
            getattr(circuit, gate.lower())(qubit)
        
        return circuit
```

### Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ --benchmark-only

# Compare with baseline
pytest tests/benchmarks/ --benchmark-compare

# Save benchmark results
pytest tests/benchmarks/ --benchmark-save=baseline

# Generate HTML report
pytest tests/benchmarks/ --benchmark-save=results --benchmark-histogram
```

## Coverage

### Running Coverage

```bash
# Generate coverage report
pytest --cov=proxima --cov-report=html

# Check coverage threshold
pytest --cov=proxima --cov-fail-under=80

# Coverage for specific module
pytest --cov=proxima.core tests/unit/core/
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = proxima
omit =
    proxima/__init__.py
    proxima/*/tests/*
    tests/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if TYPE_CHECKING:

[html]
directory = htmlcov
```

## Mocking

### Backend Mocking

```python
# tests/unit/core/test_execution_mock.py
from unittest.mock import Mock, patch
from proxima.core.execution import ExecutionEngine

class TestExecutionWithMocks:
    def test_execute_with_mock_backend(self, simple_circuit):
        mock_adapter = Mock()
        mock_adapter.execute.return_value = Mock(
            success=True,
            counts={"00": 500, "11": 500},
            shot_count=1000
        )
        
        with patch.object(
            ExecutionEngine,
            '_get_backend',
            return_value=mock_adapter
        ):
            engine = ExecutionEngine()
            result = engine.execute(simple_circuit)
        
        assert result.success
        mock_adapter.execute.assert_called_once()
```

### LLM Mocking

```python
# tests/unit/intelligence/test_llm_mock.py
from unittest.mock import Mock, patch
from proxima.intelligence.llm_router import LLMRouter

class TestLLMRouterWithMocks:
    def test_route_with_mock_provider(self):
        mock_provider = Mock()
        mock_provider.generate.return_value = Mock(
            text="Execute a Bell state circuit",
            confidence=0.95
        )
        
        router = LLMRouter(providers=[mock_provider])
        response = router.route("Create a Bell state")
        
        assert "Bell state" in response.text
```

## Test Fixtures

### Circuit Fixtures

```python
# tests/fixtures/circuits.py
import pytest
from proxima.core.circuit import Circuit

@pytest.fixture
def identity_circuit():
    """Circuit that does nothing."""
    return Circuit(1)

@pytest.fixture
def single_qubit_gates():
    """Circuit with all single-qubit gates."""
    circuit = Circuit(1)
    circuit.h(0)
    circuit.x(0)
    circuit.y(0)
    circuit.z(0)
    circuit.t(0)
    circuit.s(0)
    return circuit

@pytest.fixture
def parametrized_circuit():
    """Circuit with parametrized gates."""
    circuit = Circuit(1)
    circuit.rx(0, theta=0.5)
    circuit.ry(0, theta=1.0)
    circuit.rz(0, theta=1.5)
    return circuit
```

### Result Fixtures

```python
# tests/fixtures/results.py
import pytest
from proxima.core.result import ExecutionResult

@pytest.fixture
def bell_state_result():
    """Expected Bell state result."""
    return ExecutionResult(
        backend="lret",
        counts={"00": 500, "11": 500},
        shot_count=1000,
        qubit_count=2
    )
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest --cov=proxima --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Best Practices

### Test Naming

```python
# Good: Descriptive names
def test_execute_returns_result_with_correct_shot_count():
    pass

# Bad: Vague names
def test_execute():
    pass
```

### Arrange-Act-Assert

```python
def test_bell_state_execution():
    # Arrange
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Act
    result = execute_circuit(circuit, shots=1000)
    
    # Assert
    assert result.success
    assert "00" in result.counts or "11" in result.counts
```

### Test Independence

```python
# Good: Each test creates its own data
def test_first():
    circuit = Circuit(2)
    # ...

def test_second():
    circuit = Circuit(2)  # Fresh circuit
    # ...

# Bad: Tests share state
circuit = None

def test_first():
    global circuit
    circuit = Circuit(2)

def test_second():
    circuit.h(0)  # Depends on test_first
```

## Troubleshooting

### Common Issues

**Tests not discovered:**
```bash
# Check test naming
pytest --collect-only

# Verify conftest.py is present
ls tests/conftest.py
```

**Import errors:**
```bash
# Install in development mode
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Flaky tests:**
```bash
# Run with verbose output
pytest -v --tb=long

# Run specific test multiple times
pytest tests/path/to/test.py --count=10
```
