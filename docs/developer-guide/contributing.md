# Contributing to Proxima

Thank you for your interest in contributing to Proxima! This guide will help you get started with development and understand our contribution process.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account
- Familiarity with quantum computing concepts (helpful but not required)

### Setting Up Your Development Environment

1. **Fork the Repository:**
   - Navigate to [github.com/proxima-project/proxima](https://github.com/proxima-project/proxima)
   - Click "Fork" in the top right corner

2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/proxima.git
   cd proxima
   ```

3. **Create a Virtual Environment:**
   ```bash
   # Using venv
   python -m venv .venv
   
   # Activate (Linux/macOS)
   source .venv/bin/activate
   
   # Activate (Windows)
   .venv\Scripts\activate
   ```

4. **Install Development Dependencies:**
   ```bash
   pip install -e ".[dev,all]"
   ```

5. **Set Up Pre-commit Hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

6. **Verify Installation:**
   ```bash
   proxima --version
   pytest tests/unit -v
   ```

## Development Workflow

### 1. Create a Feature Branch

Always create a new branch for your work:

```bash
# Sync with upstream
git fetch origin
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/short-description` | `feature/add-quest-adapter` |
| Bug Fix | `fix/issue-description` | `fix/memory-leak-cuquantum` |
| Documentation | `docs/topic` | `docs/installation-guide` |
| Refactoring | `refactor/component` | `refactor/backend-registry` |
| Testing | `test/component` | `test/cirq-adapter-coverage` |

### 2. Make Your Changes

Write your code following our coding standards (see below).

### 3. Add Tests

All new features and bug fixes should include tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_my_feature.py

# Run with coverage
pytest --cov=proxima --cov-report=html
```

### 4. Run Quality Checks

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src

# Run all pre-commit hooks
pre-commit run --all-files
```

### 5. Commit Your Changes

Follow our commit message guidelines:

```bash
# Good commit messages
git commit -m "Add QuEST adapter GPU memory pooling"
git commit -m "Fix memory leak in cuQuantum batch execution"
git commit -m "Update installation docs for Windows"

# Bad commit messages (avoid)
git commit -m "Fix stuff"
git commit -m "WIP"
git commit -m "asdfgh"
```

### Commit Message Format

```
<type>: <short description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: Add QuEST adapter with GPU support

Implements the QuEST quantum simulator adapter with:
- Density matrix and state vector modes
- CUDA GPU acceleration
- OpenMP parallelization

Closes #123
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Description of what was changed and why
- Screenshots if UI changes
- Reference to related issues

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Line length: 100 characters (enforced by Black)
# Use type hints for function signatures

def execute_circuit(
    circuit: Circuit,
    backend: str = "lret",
    shots: int = 1000,
    *,
    timeout: float | None = None,
) -> ExecutionResult:
    """
    Execute a quantum circuit on the specified backend.

    Args:
        circuit: The quantum circuit to execute.
        backend: Name of the backend to use.
        shots: Number of measurement shots.
        timeout: Optional execution timeout in seconds.

    Returns:
        ExecutionResult containing measurement outcomes.

    Raises:
        BackendError: If the backend is not available.
        TimeoutError: If execution exceeds timeout.
    """
    ...
```

### Docstring Format

We use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> dict[str, Any]:
    """
    Brief one-line description.

    Longer description if needed. Can span multiple lines
    and include more details about the function.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is negative.

    Examples:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        'success'
    """
    ...
```

### Import Order

Imports are automatically sorted by `ruff`:

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party packages
import numpy as np
from pydantic import BaseModel

# Local imports
from proxima.backends import BackendRegistry
from proxima.core.executor import Executor
```

### Type Hints

Use type hints for all public functions:

```python
# Good
def process_results(
    results: list[ExecutionResult],
    threshold: float = 0.95,
) -> Summary:
    ...

# Also acceptable for complex types
from typing import TypeAlias

ResultMap: TypeAlias = dict[str, list[ExecutionResult]]

def aggregate_results(results: ResultMap) -> Summary:
    ...
```

## Testing Guidelines

### Test File Structure

```
tests/
 unit/                   # Unit tests (isolated, fast)
    test_backends.py
    test_executor.py
    test_config.py
 integration/            # Integration tests (component interaction)
    test_backend_registry.py
    test_execution_pipeline.py
 e2e/                    # End-to-end tests (full workflows)
    test_cli_e2e.py
    test_workflow_e2e.py
 conftest.py             # Shared fixtures
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from proxima.backends.cirq_adapter import CirqBackendAdapter


class TestCirqBackendAdapter:
    """Tests for the Cirq backend adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a Cirq adapter instance."""
        return CirqBackendAdapter()

    def test_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.name == "cirq"
        assert adapter.is_available()

    def test_execute_bell_circuit(self, adapter, sample_bell_circuit):
        """Test executing a Bell state circuit."""
        result = adapter.execute(sample_bell_circuit, shots=1000)
        
        assert result.success
        assert sum(result.counts.values()) == 1000
        # Bell state should have roughly equal |00 and |11
        assert 400 <= result.counts.get("00", 0) <= 600

    @pytest.mark.parametrize("shots", [100, 1000, 10000])
    def test_execute_with_different_shots(self, adapter, sample_circuit, shots):
        """Test execution with different shot counts."""
        result = adapter.execute(sample_circuit, shots=shots)
        assert sum(result.counts.values()) == shots

    @patch("proxima.backends.cirq_adapter.cirq")
    def test_handles_backend_error(self, mock_cirq, adapter):
        """Test proper error handling when backend fails."""
        mock_cirq.Simulator.side_effect = RuntimeError("Backend error")
        
        with pytest.raises(BackendError) as exc_info:
            adapter.execute(sample_circuit)
        
        assert "Backend error" in str(exc_info.value)
```

### Test Markers

```python
@pytest.mark.unit        # Fast, isolated tests
@pytest.mark.integration # Component interaction tests
@pytest.mark.e2e         # Full workflow tests
@pytest.mark.slow        # Tests that take > 1 second
@pytest.mark.backend     # Backend-specific tests
@pytest.mark.requires_network  # Tests needing network
```

## Pull Request Process

### Before Submitting

- [ ] Tests pass: `pytest`
- [ ] Code is formatted: `black src tests`
- [ ] Linting passes: `ruff check src tests`
- [ ] Type checking passes: `mypy src`
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for user-facing changes

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review
- [ ] I have added tests for my changes
- [ ] All new and existing tests pass
- [ ] I have updated documentation as needed
```

### Code Review

- Address reviewer comments promptly
- Push fixes as new commits (don't force-push during review)
- Request re-review after addressing comments
- Be respectful and open to feedback

## Adding New Features

### Adding a New Backend

See our dedicated guide: [Adding Backends](adding-backends.md)

### Adding a CLI Command

1. Create command file in `src/proxima/cli/commands/`
2. Register in `src/proxima/cli/main.py`
3. Add tests in `tests/unit/test_cli.py`
4. Update CLI reference documentation

### Adding Configuration Options

1. Add to `src/proxima/config/settings.py`
2. Add defaults in `src/proxima/config/defaults.py`
3. Update configuration documentation
4. Add validation tests

## Documentation

### Building Docs Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build and serve docs
mkdocs serve

# View at http://localhost:8000
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep examples up-to-date with code

## Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open a GitHub Issue with reproduction steps
- **Feature Requests:** Open a GitHub Issue with use case description
- **Security Issues:** Email security@proxima-project.io (do not open public issues)

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor graph

Thank you for contributing to Proxima! 
