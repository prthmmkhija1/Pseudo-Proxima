"""
E2E Tests for Agent Interpreter and Built-in Executors.

Comprehensive end-to-end tests covering:
- Agent file parsing and execution
- Built-in executor workflows
- Complex task orchestration
- Error recovery scenarios
- Consent management integration
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import asyncio

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for E2E tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "agents").mkdir()
        (workspace / "circuits").mkdir()
        (workspace / "results").mkdir()
        yield workspace


@pytest.fixture
def sample_agent_content():
    """Sample agent file content for testing."""
    return """---
name: Test Agent
version: 1.0.0
description: Test agent for E2E testing
author: Test Suite
requires_consent: false
trusted: true
---

## Configuration

```yaml
default_backend: auto
backends:
  - cirq
  - qiskit_aer
shots: 1024
timeout_seconds: 300
continue_on_error: true
parallel_execution: false
```

## Tasks

### Task: create_bell_state

Execute a Bell state circuit on the default backend.

```yaml
type: circuit_execution
parameters:
  backend: auto
  shots: 1024
```

### Task: analyze_results

Analyze the results from the previous task.

```yaml
type: result_analysis
depends_on:
  - create_bell_state
parameters:
  detail_level: standard
```
"""


@pytest.fixture
def complex_agent_content():
    """Complex agent file with multiple task types."""
    return """---
name: Complex Test Agent
version: 2.0.0
description: Complex agent with multiple task types
author: Test Suite
---

## Configuration

```yaml
default_backend: auto
shots: 512
timeout_seconds: 600
continue_on_error: true
```

## Tasks

### Task: benchmark_backends

Benchmark multiple backends.

```yaml
type: benchmark
parameters:
  backends:
    - cirq
    - qiskit_aer
  shots_list:
    - 100
    - 512
    - 1024
```

### Task: optimize_circuit

Optimize the circuit.

```yaml
type: optimization
parameters:
  optimization_level: 2
  optimization_goals:
    - depth
    - gates
```

### Task: visualize_results

Visualize the benchmark results.

```yaml
type: visualization
depends_on:
  - benchmark_backends
parameters:
  visualization_type: comparison
  title: Backend Comparison
```

### Task: noise_analysis

Analyze noise characteristics.

```yaml
type: noise_analysis
parameters:
  analysis_type: quick
```

### Task: batch_circuits

Execute batch of circuits.

```yaml
type: batch_execution
parameters:
  backend: auto
  shots: 256
```
"""


# =============================================================================
# Agent File Parsing Tests
# =============================================================================


class TestAgentFileParsing:
    """Tests for agent file parsing."""

    def test_parse_simple_agent_file(self, temp_workspace, sample_agent_content):
        """Test parsing a simple agent file."""
        from proxima.core.agent_interpreter import AgentFileParser
        
        agent_file = temp_workspace / "agents" / "test_agent.md"
        agent_file.write_text(sample_agent_content)
        
        parser = AgentFileParser()
        agent = parser.parse_file(agent_file)
        
        assert agent.metadata.name == "Test Agent"
        assert agent.metadata.version == "1.0.0"
        assert agent.is_valid
        assert len(agent.tasks) >= 1

    def test_parse_complex_agent_file(self, temp_workspace, complex_agent_content):
        """Test parsing a complex agent file with multiple task types."""
        from proxima.core.agent_interpreter import AgentFileParser
        
        agent_file = temp_workspace / "agents" / "complex_agent.md"
        agent_file.write_text(complex_agent_content)
        
        parser = AgentFileParser()
        agent = parser.parse_file(agent_file)
        
        assert agent.metadata.name == "Complex Test Agent"
        assert agent.is_valid
        assert len(agent.tasks) >= 3

    def test_parse_invalid_agent_file(self, temp_workspace):
        """Test parsing an invalid agent file."""
        from proxima.core.agent_interpreter import AgentFileParser
        
        agent_file = temp_workspace / "agents" / "invalid_agent.md"
        agent_file.write_text("This is not a valid agent file")
        
        parser = AgentFileParser()
        agent = parser.parse_content("This is not a valid agent file")
        
        # Should parse without crashing
        assert agent is not None

    def test_parse_agent_with_dependencies(self, temp_workspace):
        """Test parsing agent file with task dependencies."""
        content = """---
name: Dependency Test
version: 1.0.0
---

## Tasks

### Task: first_task
```yaml
type: circuit_execution
```

### Task: second_task
```yaml
type: result_analysis
depends_on:
  - first_task
```

### Task: third_task
```yaml
type: export
depends_on:
  - second_task
```
"""
        from proxima.core.agent_interpreter import AgentFileParser
        
        parser = AgentFileParser()
        agent = parser.parse_content(content)
        
        assert agent is not None
        # Tasks should preserve dependency information
        for task in agent.tasks:
            if task.id == "second_task":
                assert "first_task" in task.depends_on


# =============================================================================
# Built-in Executor Tests
# =============================================================================


class TestBuiltinExecutors:
    """Tests for built-in executors."""

    def test_benchmark_executor_creation(self):
        """Test BenchmarkExecutor can be created."""
        from proxima.core.agent_interpreter import BenchmarkExecutor
        
        executor = BenchmarkExecutor(
            warmup_runs=2,
            benchmark_runs=5,
            collect_memory=True,
        )
        
        assert executor.warmup_runs == 2
        assert executor.benchmark_runs == 5
        assert executor.collect_memory is True

    def test_optimization_executor_creation(self):
        """Test OptimizationExecutor can be created."""
        from proxima.core.agent_interpreter import OptimizationExecutor
        
        executor = OptimizationExecutor(optimization_level=3)
        
        assert executor.optimization_level == 3

    def test_visualization_executor_creation(self):
        """Test VisualizationExecutor can be created."""
        from proxima.core.agent_interpreter import VisualizationExecutor
        
        executor = VisualizationExecutor(output_format="html")
        
        assert executor.output_format == "html"

    def test_state_tomography_executor_creation(self):
        """Test StateTomographyExecutor can be created."""
        from proxima.core.agent_interpreter import StateTomographyExecutor
        
        executor = StateTomographyExecutor(method="mle")
        
        assert executor.method == "mle"

    def test_noise_analysis_executor_creation(self):
        """Test NoiseAnalysisExecutor can be created."""
        from proxima.core.agent_interpreter import NoiseAnalysisExecutor
        
        executor = NoiseAnalysisExecutor()
        assert executor is not None

    def test_batch_executor_creation(self):
        """Test BatchExecutor can be created."""
        from proxima.core.agent_interpreter import BatchExecutor
        
        executor = BatchExecutor(max_parallel=8, continue_on_error=False)
        
        assert executor.max_parallel == 8
        assert executor.continue_on_error is False

    def test_get_builtin_executor(self):
        """Test get_builtin_executor factory function."""
        from proxima.core.agent_interpreter import (
            get_builtin_executor,
            BUILTIN_EXECUTORS,
        )
        
        for name in BUILTIN_EXECUTORS:
            executor = get_builtin_executor(name)
            assert executor is not None

    def test_get_builtin_executor_unknown(self):
        """Test get_builtin_executor with unknown executor."""
        from proxima.core.agent_interpreter import get_builtin_executor
        
        with pytest.raises(ValueError, match="Unknown built-in executor"):
            get_builtin_executor("unknown_executor")


# =============================================================================
# Executor Execution Tests
# =============================================================================


class TestExecutorExecution:
    """Tests for executor execution with mocked backends."""

    def test_benchmark_executor_execute(self):
        """Test BenchmarkExecutor execution."""
        from proxima.core.agent_interpreter import (
            BenchmarkExecutor,
            TaskDefinition,
            TaskType,
        )
        
        executor = BenchmarkExecutor(warmup_runs=1, benchmark_runs=2)
        
        # Create mock task
        task = TaskDefinition(
            id="test_benchmark",
            name="Test Benchmark",
            type=TaskType.BENCHMARK,
            parameters={
                "backends": [],  # Empty to test graceful handling
                "shots_list": [100],
            },
        )
        
        with patch("proxima.backends.registry.backend_registry") as mock_registry:
            mock_registry.list_backends.return_value = []
            result = executor.execute(task, {})
        
        assert result["task_type"] == "benchmark"
        assert "backends" in result
        assert "summary" in result

    def test_optimization_executor_execute(self):
        """Test OptimizationExecutor execution."""
        from proxima.core.agent_interpreter import (
            OptimizationExecutor,
            TaskDefinition,
            TaskType,
        )
        
        executor = OptimizationExecutor(optimization_level=2)
        
        # Create mock circuit
        mock_circuit = Mock()
        mock_circuit.num_qubits = 4
        mock_circuit.gate_count = 20
        mock_circuit.depth = 10
        
        task = TaskDefinition(
            id="test_optimization",
            name="Test Optimization",
            type=TaskType.OPTIMIZATION,
            parameters={
                "circuit": mock_circuit,
                "optimization_goals": ["depth", "gates"],
            },
        )
        
        result = executor.execute(task, {})
        
        assert result["task_type"] == "optimization"
        assert result["success"] is True
        assert "original_metrics" in result
        assert "optimized_metrics" in result
        assert "improvement" in result
        assert "passes_applied" in result

    def test_visualization_executor_execute(self):
        """Test VisualizationExecutor execution."""
        from proxima.core.agent_interpreter import (
            VisualizationExecutor,
            TaskDefinition,
            TaskType,
        )
        
        executor = VisualizationExecutor(output_format="text")
        
        task = TaskDefinition(
            id="test_viz",
            name="Test Visualization",
            type=TaskType.VISUALIZATION,
            parameters={
                "visualization_type": "histogram",
                "data": {"counts": {"00": 500, "11": 500}},
                "title": "Test Histogram",
            },
        )
        
        result = executor.execute(task, {})
        
        assert result["task_type"] == "visualization"
        assert result["success"] is True
        assert "output" in result

    def test_noise_analysis_executor_execute(self):
        """Test NoiseAnalysisExecutor execution."""
        from proxima.core.agent_interpreter import (
            NoiseAnalysisExecutor,
            TaskDefinition,
            TaskType,
        )
        
        executor = NoiseAnalysisExecutor()
        
        mock_circuit = Mock()
        mock_circuit.num_qubits = 4
        mock_circuit.gate_count = 15
        
        task = TaskDefinition(
            id="test_noise",
            name="Test Noise Analysis",
            type=TaskType.NOISE_ANALYSIS,
            parameters={
                "circuit": mock_circuit,
                "backend": "auto",
                "analysis_type": "quick",
            },
        )
        
        result = executor.execute(task, {})
        
        assert result["task_type"] == "noise_analysis"
        assert result["success"] is True
        assert "noise_parameters" in result
        assert "error_budget" in result
        assert "recommendations" in result

    def test_state_tomography_executor_execute(self):
        """Test StateTomographyExecutor execution."""
        from proxima.core.agent_interpreter import (
            StateTomographyExecutor,
            TaskDefinition,
            TaskType,
        )
        
        executor = StateTomographyExecutor(method="linear_inversion")
        
        mock_circuit = Mock()
        mock_circuit.num_qubits = 2
        
        task = TaskDefinition(
            id="test_tomography",
            name="Test Tomography",
            type=TaskType.STATE_TOMOGRAPHY,
            parameters={
                "circuit": mock_circuit,
                "shots_per_basis": 512,
            },
        )
        
        result = executor.execute(task, {})
        
        assert result["task_type"] == "state_tomography"
        assert result["success"] is True
        assert "reconstructed_state" in result
        assert "confidence" in result


# =============================================================================
# Agent Interpreter Integration Tests
# =============================================================================


class TestAgentInterpreterIntegration:
    """Integration tests for AgentInterpreter."""

    def test_interpreter_creation(self):
        """Test AgentInterpreter can be created."""
        from proxima.core.agent_interpreter import AgentInterpreter
        
        interpreter = AgentInterpreter()
        
        assert interpreter is not None

    def test_interpreter_with_custom_executor(self):
        """Test AgentInterpreter with custom executor."""
        from proxima.core.agent_interpreter import (
            AgentInterpreter,
            DefaultTaskExecutor,
        )
        
        custom_executor = DefaultTaskExecutor(
            default_shots=512,
            default_backend="cirq",
        )
        
        interpreter = AgentInterpreter(executor=custom_executor)
        
        assert interpreter.executor == custom_executor

    def test_interpreter_with_progress_callback(self):
        """Test AgentInterpreter with progress callback."""
        from proxima.core.agent_interpreter import AgentInterpreter
        
        progress_updates = []
        
        def progress_callback(message: str, progress: float):
            progress_updates.append((message, progress))
        
        interpreter = AgentInterpreter(progress_callback=progress_callback)
        
        assert interpreter is not None


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery mechanisms."""

    def test_recovery_strategy_creation(self):
        """Test recovery strategy creation."""
        from proxima.core.agent_interpreter import (
            RetryStrategy,
            FallbackStrategy,
            SkipStrategy,
        )
        
        retry = RetryStrategy(max_retries=3, delay_seconds=1.0)
        fallback = FallbackStrategy(fallback_value=None)
        skip = SkipStrategy()
        
        assert retry.max_retries == 3
        assert fallback.fallback_value is None
        assert skip is not None

    def test_error_recovery_manager_creation(self):
        """Test ErrorRecoveryManager creation."""
        from proxima.core.agent_interpreter import (
            ErrorRecoveryManager,
            RetryStrategy,
        )
        
        manager = ErrorRecoveryManager()
        manager.register_strategy("test", RetryStrategy(max_retries=2))
        
        assert manager is not None


# =============================================================================
# Task Type Tests
# =============================================================================


class TestTaskTypes:
    """Tests for all task types."""

    def test_all_task_types_exist(self):
        """Test all task types are defined."""
        from proxima.core.agent_interpreter import TaskType
        
        expected_types = [
            "CIRCUIT_EXECUTION",
            "BACKEND_COMPARISON",
            "RESULT_ANALYSIS",
            "EXPORT",
            "CUSTOM",
            "BENCHMARK",
            "OPTIMIZATION",
            "VISUALIZATION",
            "STATE_TOMOGRAPHY",
            "NOISE_ANALYSIS",
            "BATCH_EXECUTION",
        ]
        
        for type_name in expected_types:
            assert hasattr(TaskType, type_name), f"TaskType.{type_name} not found"

    def test_task_type_values(self):
        """Test task type enum values."""
        from proxima.core.agent_interpreter import TaskType
        
        assert TaskType.CIRCUIT_EXECUTION.value == "circuit_execution"
        assert TaskType.BENCHMARK.value == "benchmark"
        assert TaskType.OPTIMIZATION.value == "optimization"
        assert TaskType.VISUALIZATION.value == "visualization"
        assert TaskType.STATE_TOMOGRAPHY.value == "state_tomography"
        assert TaskType.NOISE_ANALYSIS.value == "noise_analysis"
        assert TaskType.BATCH_EXECUTION.value == "batch_execution"
