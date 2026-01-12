"""Step 5.1: Unit Testing - Backend Selection Tests.

Comprehensive test suite for unified backend selection covering:
- Auto-selection algorithm
- GPU-aware selection
- Priority-based selection
- Fallback logic

Test Categories:
| Test Type       | Purpose                                       |
|-----------------|-----------------------------------------------|
| Auto-Selection  | Intelligent backend auto-selection            |
| GPU Selection   | GPU-aware backend selection                   |
| Priority        | Priority-based selection strategies           |
| Fallback        | Fallback when preferred backend unavailable   |
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_backend_registry():
    """Mock backend registry with all backends."""
    registry = MagicMock()

    # Available backends
    registry.list_available.return_value = [
        "lret",
        "cirq",
        "qiskit",
        "quest",
        "cuquantum",
        "qsim",
    ]

    # Backend capabilities
    registry.get_capabilities.side_effect = lambda name: {
        "lret": {"max_qubits": 15, "supports_gpu": False, "supports_dm": True},
        "cirq": {"max_qubits": 20, "supports_gpu": False, "supports_dm": True},
        "qiskit": {"max_qubits": 30, "supports_gpu": False, "supports_dm": True},
        "quest": {"max_qubits": 30, "supports_gpu": True, "supports_dm": True},
        "cuquantum": {"max_qubits": 35, "supports_gpu": True, "supports_dm": False},
        "qsim": {"max_qubits": 35, "supports_gpu": False, "supports_dm": False},
    }.get(name, {})

    return registry


@pytest.fixture
def mock_gpu_available():
    """Mock GPU as available."""
    with patch("proxima.intelligence.selector.is_gpu_available") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU as unavailable."""
    with patch("proxima.intelligence.selector.is_gpu_available") as mock:
        mock.return_value = False
        yield mock


@pytest.fixture
def small_circuit():
    """Small circuit (5 qubits)."""
    return {
        "num_qubits": 5,
        "gates": [{"name": "H", "qubits": [i]} for i in range(5)],
        "simulator_type": "state_vector",
    }


@pytest.fixture
def medium_circuit():
    """Medium circuit (15 qubits)."""
    return {
        "num_qubits": 15,
        "gates": [{"name": "H", "qubits": [i]} for i in range(15)],
        "simulator_type": "state_vector",
    }


@pytest.fixture
def large_circuit():
    """Large circuit (25 qubits)."""
    return {
        "num_qubits": 25,
        "gates": [{"name": "H", "qubits": [i]} for i in range(25)],
        "simulator_type": "state_vector",
    }


@pytest.fixture
def density_matrix_circuit():
    """Circuit requiring density matrix simulation."""
    return {
        "num_qubits": 10,
        "gates": [{"name": "H", "qubits": [i]} for i in range(10)],
        "simulator_type": "density_matrix",
    }


@pytest.fixture
def noisy_circuit():
    """Circuit with noise model."""
    return {
        "num_qubits": 8,
        "gates": [{"name": "H", "qubits": [i]} for i in range(8)],
        "simulator_type": "density_matrix",
        "noise_model": {"depolarizing": 0.01},
    }


# =============================================================================
# STEP 5.2.1: BACKEND REGISTRY TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestBackendRegistry:
    """Tests for backend registry functionality."""

    def test_all_backends_discovered(self, mock_backend_registry):
        """Test that all backends are discovered."""
        backends = mock_backend_registry.list_available()

        assert "lret" in backends
        assert "cirq" in backends
        assert "qiskit" in backends
        assert "quest" in backends
        assert "cuquantum" in backends
        assert "qsim" in backends

    def test_backend_capabilities_retrieval(self, mock_backend_registry):
        """Test capability retrieval for each backend."""
        for backend in ["lret", "cirq", "qiskit", "quest", "cuquantum", "qsim"]:
            caps = mock_backend_registry.get_capabilities(backend)

            assert "max_qubits" in caps
            assert isinstance(caps["max_qubits"], int)

    def test_gpu_backends_identified(self, mock_backend_registry):
        """Test that GPU backends are correctly identified."""
        gpu_backends = []
        for backend in mock_backend_registry.list_available():
            caps = mock_backend_registry.get_capabilities(backend)
            if caps.get("supports_gpu"):
                gpu_backends.append(backend)

        assert "quest" in gpu_backends
        assert "cuquantum" in gpu_backends
        assert "cirq" not in gpu_backends

    def test_density_matrix_backends_identified(self, mock_backend_registry):
        """Test that DM backends are correctly identified."""
        dm_backends = []
        for backend in mock_backend_registry.list_available():
            caps = mock_backend_registry.get_capabilities(backend)
            if caps.get("supports_dm"):
                dm_backends.append(backend)

        assert "quest" in dm_backends
        assert "cirq" in dm_backends
        assert "cuquantum" not in dm_backends  # SV only


# =============================================================================
# STEP 5.2.2: AUTO-SELECTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestBackendAutoSelection:
    """Tests for automatic backend selection."""

    def test_auto_select_small_circuit(self, mock_backend_registry, small_circuit):
        """Test auto-selection for small circuits."""
        # Small circuits can use any backend
        # Selector should choose based on other criteria
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()
            result = selector.select(small_circuit)

            assert result is not None
            assert result in mock_backend_registry.list_available()

    def test_auto_select_large_circuit(self, mock_backend_registry, large_circuit):
        """Test auto-selection for large circuits."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()
            result = selector.select(large_circuit)

            # Should select backend with high max_qubits
            assert result is not None
            caps = mock_backend_registry.get_capabilities(result)
            assert caps["max_qubits"] >= 25

    def test_auto_select_density_matrix(
        self, mock_backend_registry, density_matrix_circuit
    ):
        """Test auto-selection for density matrix circuits."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()
            result = selector.select(density_matrix_circuit)

            # Should select DM-capable backend
            caps = mock_backend_registry.get_capabilities(result)
            assert caps.get("supports_dm") is True

    def test_auto_select_noisy_circuit(self, mock_backend_registry, noisy_circuit):
        """Test auto-selection for noisy circuits."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()
            result = selector.select(noisy_circuit)

            # Should select noise-supporting backend
            assert result in ["quest", "qiskit", "cirq"]

    def test_selection_explanation_provided(self, mock_backend_registry, small_circuit):
        """Test that selection includes explanation."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()
            result, explanation = selector.select_with_explanation(small_circuit)

            assert result is not None
            assert explanation is not None
            assert isinstance(explanation, str)


# =============================================================================
# STEP 5.2.3: GPU-AWARE SELECTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestGPUAwareSelection:
    """Tests for GPU-aware backend selection."""

    def test_prefer_gpu_when_available(
        self, mock_backend_registry, large_circuit, mock_gpu_available
    ):
        """Test that GPU backends are preferred when GPU is available."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector(strategy="gpu_preferred")
            result = selector.select(large_circuit)

            # Should prefer GPU backend
            caps = mock_backend_registry.get_capabilities(result)
            assert caps.get("supports_gpu") is True

    def test_fallback_to_cpu_when_no_gpu(
        self, mock_backend_registry, large_circuit, mock_gpu_unavailable
    ):
        """Test fallback to CPU when GPU is unavailable."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector(strategy="gpu_preferred")
            result = selector.select(large_circuit)

            # Should fall back to CPU backend
            assert result is not None
            # qsim is CPU-optimized for large SV
            assert result in ["qsim", "quest", "qiskit"]

    def test_cpu_optimized_strategy(self, mock_backend_registry, large_circuit):
        """Test CPU-optimized selection strategy."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector(strategy="cpu_optimized")
            result = selector.select(large_circuit)

            # Should prefer CPU-optimized backends
            assert result in ["qsim", "quest"]

    def test_gpu_memory_consideration(self, mock_backend_registry, mock_gpu_available):
        """Test that GPU memory is considered in selection."""
        huge_circuit = {
            "num_qubits": 32,  # Would need ~64GB GPU memory
            "gates": [],
            "simulator_type": "state_vector",
        }

        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector(strategy="gpu_preferred")

            # Should either select or warn about GPU memory
            try:
                result = selector.select(huge_circuit)
                # If selection succeeds, it should handle this case
                assert result is not None
            except Exception as e:
                # May raise memory warning
                assert "memory" in str(e).lower() or "qubit" in str(e).lower()


# =============================================================================
# STEP 5.2.4: PRIORITY-BASED SELECTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestPriorityBasedSelection:
    """Tests for priority-based backend selection."""

    def test_state_vector_gpu_priority(self, mock_backend_registry, mock_gpu_available):
        """Test priority for state vector with GPU."""
        from proxima.intelligence.selector import BackendPrioritySelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendPrioritySelector()
            priorities = selector.get_priority_list("state_vector_gpu")

            assert "cuquantum" in priorities[:2]
            assert "quest" in priorities[:3]

    def test_state_vector_cpu_priority(self, mock_backend_registry):
        """Test priority for state vector with CPU."""
        from proxima.intelligence.selector import BackendPrioritySelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendPrioritySelector()
            priorities = selector.get_priority_list("state_vector_cpu")

            assert "qsim" in priorities[:2]
            assert "quest" in priorities[:3]

    def test_density_matrix_priority(self, mock_backend_registry):
        """Test priority for density matrix simulations."""
        from proxima.intelligence.selector import BackendPrioritySelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendPrioritySelector()
            priorities = selector.get_priority_list("density_matrix")

            assert "quest" in priorities[:2]
            assert "cirq" in priorities[:3]

    def test_noisy_circuit_priority(self, mock_backend_registry):
        """Test priority for noisy circuits."""
        from proxima.intelligence.selector import BackendPrioritySelector

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendPrioritySelector()
            priorities = selector.get_priority_list("noisy_circuit")

            assert "quest" in priorities[:3]
            assert "qiskit" in priorities[:3]

    def test_first_available_selection(self, mock_backend_registry, small_circuit):
        """Test selection of first available backend from priority list."""
        from proxima.intelligence.selector import BackendPrioritySelector

        # Mock some backends as unavailable
        mock_backend_registry.is_available = lambda name: name != "cuquantum"

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendPrioritySelector()
            result = selector.select_first_available(
                ["cuquantum", "quest", "qsim"], small_circuit
            )

            # Should skip cuquantum and select quest
            assert result == "quest"


# =============================================================================
# STEP 5.2.5: FALLBACK LOGIC TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestFallbackLogic:
    """Tests for backend fallback logic."""

    def test_fallback_on_backend_unavailable(
        self, mock_backend_registry, small_circuit
    ):
        """Test fallback when preferred backend is unavailable."""
        from proxima.intelligence.selector import BackendSelector

        # Make preferred backend unavailable
        mock_backend_registry.is_available = lambda name: name != "quest"

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector(preferred="quest")
            result = selector.select(small_circuit)

            # Should fall back to another backend
            assert result is not None
            assert result != "quest"

    def test_fallback_on_validation_failure(self, mock_backend_registry):
        """Test fallback when circuit validation fails."""
        from proxima.intelligence.selector import BackendSelector

        # Circuit that exceeds lret's qubit limit
        circuit = {"num_qubits": 20, "gates": [], "simulator_type": "state_vector"}

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector(preferred="lret")  # max 15 qubits
            result = selector.select(circuit)

            # Should fall back to backend with higher qubit limit
            caps = mock_backend_registry.get_capabilities(result)
            assert caps["max_qubits"] >= 20

    def test_fallback_chain(self, mock_backend_registry, medium_circuit):
        """Test fallback chain through multiple backends."""
        from proxima.intelligence.selector import BackendSelector

        # Make multiple backends unavailable
        mock_backend_registry.is_available = lambda name: name in ["cirq", "qiskit"]

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()
            result = selector.select(medium_circuit)

            assert result in ["cirq", "qiskit"]

    def test_no_suitable_backend_error(self, mock_backend_registry):
        """Test error when no suitable backend is available."""
        from proxima.intelligence.selector import BackendSelector

        # Make all backends unavailable
        mock_backend_registry.list_available.return_value = []

        with patch(
            "proxima.intelligence.selector.BackendRegistry",
            return_value=mock_backend_registry,
        ):
            selector = BackendSelector()

            with pytest.raises(ValueError):
                selector.select({"num_qubits": 5, "gates": []})


# =============================================================================
# COMPARISON MATRIX TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestBackendComparisonMatrix:
    """Tests for backend comparison matrix."""

    def test_comparison_matrix_structure(self):
        """Test comparison matrix structure."""
        from proxima.data.compare import BackendComparisonMatrix

        matrix = BackendComparisonMatrix()

        # Should have entries for all backends
        assert "lret" in matrix.MATRIX
        assert "cirq" in matrix.MATRIX
        assert "quest" in matrix.MATRIX
        assert "cuquantum" in matrix.MATRIX
        assert "qsim" in matrix.MATRIX

    def test_get_recommendation(self):
        """Test recommendation based on requirements."""
        from proxima.data.compare import BackendComparisonMatrix

        matrix = BackendComparisonMatrix()

        # Get recommendation for GPU + large SV
        rec = matrix.get_recommendation(
            requires_gpu=True,
            requires_dm=False,
            min_qubits=30,
        )

        assert rec in ["cuquantum", "quest"]

    def test_compare_two_backends(self):
        """Test comparison of two specific backends."""
        from proxima.data.compare import BackendComparisonMatrix

        matrix = BackendComparisonMatrix()

        comparison = matrix.compare("quest", "qsim")

        assert "quest" in comparison
        assert "qsim" in comparison

    def test_to_markdown_table(self):
        """Test markdown table generation."""
        from proxima.data.compare import BackendComparisonMatrix

        matrix = BackendComparisonMatrix()

        table = matrix.to_markdown_table()

        assert isinstance(table, str)
        assert "Backend" in table
        assert "|" in table  # Markdown table format
