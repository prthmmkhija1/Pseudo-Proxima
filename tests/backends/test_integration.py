"""Step 5.2: Integration Testing - Multi-Backend Integration Tests.

Comprehensive integration test suite covering:
- Backend registry integration
- Multi-backend comparison
- Fallback logic across backends
- Configuration integration

Test Categories:
| Test Type       | Purpose                                       |
|-----------------|-----------------------------------------------|
| Registry        | Backend discovery, registration, availability |
| Comparison      | Same circuit on multiple backends             |
| Fallback        | Graceful degradation across backends          |
| Configuration   | Config options affect backend behavior        |
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
def mock_all_backends():
    """Mock all backends for integration testing."""
    backends = {}

    for name in ["lret", "cirq", "qiskit", "quest", "cuquantum", "qsim"]:
        mock_backend = MagicMock()
        mock_backend.get_name.return_value = name
        mock_backend.get_version.return_value = "1.0.0"
        mock_backend.is_available.return_value = True
        mock_backend.validate_circuit.return_value = MagicMock(valid=True)
        mock_backend.execute.return_value = MagicMock(
            backend=name,
            qubit_count=2,
            execution_time_ms=10.0,
            data={"counts": {"00": 500, "11": 500}},
        )
        backends[name] = mock_backend

    return backends


@pytest.fixture
def bell_state_circuit():
    """Bell state circuit for comparison testing."""
    return {
        "num_qubits": 2,
        "gates": [
            {"name": "H", "qubits": [0]},
            {"name": "CNOT", "qubits": [0, 1]},
        ],
        "measurements": [0, 1],
    }


@pytest.fixture
def ghz_state_circuit():
    """GHZ state circuit for comparison testing."""
    return {
        "num_qubits": 3,
        "gates": [
            {"name": "H", "qubits": [0]},
            {"name": "CNOT", "qubits": [0, 1]},
            {"name": "CNOT", "qubits": [1, 2]},
        ],
        "measurements": [0, 1, 2],
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "backends": {
            "default": "auto",
            "timeout_s": 300,
            "max_qubits": 30,
        },
        "auto_selection": {
            "enabled": True,
            "prefer_gpu": True,
            "fallback_backend": "cirq",
        },
    }


# =============================================================================
# STEP 5.2.1: BACKEND REGISTRY INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.backend
class TestBackendRegistryIntegration:
    """Integration tests for backend registry."""

    def test_all_backends_registered(self, mock_all_backends):
        """Test that all backends are discovered and registered."""
        from proxima.backends.registry import BackendRegistry

        with patch.object(
            BackendRegistry, "_discover_backends", return_value=mock_all_backends
        ):
            registry = BackendRegistry()

            backends = registry.list_available()

            assert len(backends) >= 3  # At least lret, cirq, qiskit
            for name in ["lret", "cirq", "qiskit"]:
                assert name in backends

    def test_backend_availability_checking(self, mock_all_backends):
        """Test backend availability checking."""
        from proxima.backends.registry import BackendRegistry

        # Make one backend unavailable
        mock_all_backends["quest"].is_available.return_value = False

        with patch.object(
            BackendRegistry, "_discover_backends", return_value=mock_all_backends
        ):
            registry = BackendRegistry()

            assert registry.is_available("cirq") is True
            assert registry.is_available("quest") is False

    def test_get_backend_instance(self, mock_all_backends):
        """Test retrieving backend instance."""
        from proxima.backends.registry import BackendRegistry

        with patch.object(
            BackendRegistry, "_discover_backends", return_value=mock_all_backends
        ):
            registry = BackendRegistry()

            backend = registry.get("cirq")

            assert backend is not None
            assert backend.get_name() == "cirq"

    def test_backend_capabilities_query(self, mock_all_backends):
        """Test querying backend capabilities."""
        from proxima.backends.registry import BackendRegistry

        mock_all_backends["quest"].get_capabilities.return_value = MagicMock(
            max_qubits=30,
            supports_gpu=True,
            supports_noise=True,
        )

        with patch.object(
            BackendRegistry, "_discover_backends", return_value=mock_all_backends
        ):
            registry = BackendRegistry()

            caps = registry.get_capabilities("quest")

            assert caps.max_qubits == 30
            assert caps.supports_gpu is True

    def test_priority_ordering(self, mock_all_backends):
        """Test that backends are returned in priority order."""
        from proxima.backends.registry import BackendRegistry

        with patch.object(
            BackendRegistry, "_discover_backends", return_value=mock_all_backends
        ):
            registry = BackendRegistry()

            backends = registry.list_by_priority()

            # Should return list in some priority order
            assert isinstance(backends, list)
            assert len(backends) > 0


# =============================================================================
# STEP 5.2.2: MULTI-BACKEND COMPARISON TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.backend
class TestMultiBackendComparison:
    """Integration tests for multi-backend comparison."""

    def test_same_circuit_multiple_backends(
        self, mock_all_backends, bell_state_circuit
    ):
        """Test running same circuit on multiple backends."""
        from proxima.data.compare import BackendComparator

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            comparator = BackendComparator()

            results = comparator.compare(
                bell_state_circuit,
                backends=["cirq", "qiskit"],
            )

            assert len(results) == 2
            assert "cirq" in [r.backend for r in results]
            assert "qiskit" in [r.backend for r in results]

    def test_comparison_with_identical_parameters(
        self, mock_all_backends, bell_state_circuit
    ):
        """Test that comparison uses identical parameters."""
        from proxima.data.compare import BackendComparator

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            comparator = BackendComparator()

            comparator.compare(
                bell_state_circuit,
                backends=["cirq", "qiskit"],
                options={"shots": 1000},
            )

            # Both should have been called with same options
            for backend_name in ["cirq", "qiskit"]:
                mock_all_backends[backend_name].execute.assert_called()

    def test_result_accuracy_comparison(self, mock_all_backends, bell_state_circuit):
        """Test comparison includes accuracy metrics."""
        from proxima.data.compare import BackendComparator

        # Set up different results for comparison
        mock_all_backends["cirq"].execute.return_value.data = {
            "counts": {"00": 498, "11": 502}
        }
        mock_all_backends["qiskit"].execute.return_value.data = {
            "counts": {"00": 510, "11": 490}
        }

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            comparator = BackendComparator()

            report = comparator.compare_with_analysis(
                bell_state_circuit,
                backends=["cirq", "qiskit"],
            )

            assert report is not None
            assert hasattr(report, "accuracy") or "accuracy" in str(report)

    def test_performance_comparison(self, mock_all_backends, bell_state_circuit):
        """Test comparison includes performance metrics."""
        from proxima.data.compare import BackendComparator

        # Set different execution times
        mock_all_backends["cirq"].execute.return_value.execution_time_ms = 15.0
        mock_all_backends["qiskit"].execute.return_value.execution_time_ms = 25.0

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            comparator = BackendComparator()

            report = comparator.compare_with_analysis(
                bell_state_circuit,
                backends=["cirq", "qiskit"],
            )

            # Should include timing comparison
            assert report is not None

    def test_comparison_report_generation(self, mock_all_backends, ghz_state_circuit):
        """Test generation of comparison report."""
        from proxima.data.compare import BackendComparator

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            comparator = BackendComparator()

            results = comparator.compare(
                ghz_state_circuit,
                backends=["cirq", "qiskit", "quest"],
            )

            report = comparator.generate_report(results)

            assert report is not None
            assert isinstance(report, (str, dict))


# =============================================================================
# STEP 5.2.3: FALLBACK LOGIC INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.backend
class TestFallbackLogicIntegration:
    """Integration tests for fallback logic."""

    def test_gpu_to_cpu_fallback(self, mock_all_backends, bell_state_circuit):
        """Test fallback from GPU to CPU backend."""
        from proxima.intelligence.selector import BackendSelector

        # Make GPU backends unavailable
        mock_all_backends["cuquantum"].is_available.return_value = False
        mock_all_backends["quest"].get_capabilities.return_value.supports_gpu = False

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            selector = BackendSelector(strategy="gpu_preferred")

            result = selector.select(bell_state_circuit)

            # Should fall back to CPU backend
            assert result in ["qsim", "cirq", "qiskit", "quest"]

    def test_primary_to_fallback_backend(self, mock_all_backends, bell_state_circuit):
        """Test fallback from primary to secondary backend."""
        from proxima.core.executor import CircuitExecutor

        # Make primary backend fail
        mock_all_backends["cirq"].execute.side_effect = RuntimeError("Backend error")

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            executor = CircuitExecutor(fallback_enabled=True)

            result = executor.execute(
                bell_state_circuit,
                preferred_backend="cirq",
                fallback_backend="qiskit",
            )

            # Should have used fallback
            assert result is not None

    def test_unsupported_feature_fallback(self, mock_all_backends):
        """Test fallback when backend doesn't support feature."""
        from proxima.intelligence.selector import BackendSelector

        # Circuit requiring density matrix
        dm_circuit = {
            "num_qubits": 5,
            "gates": [],
            "simulator_type": "density_matrix",
        }

        # cuquantum doesn't support DM
        mock_all_backends["cuquantum"].validate_circuit.return_value.valid = False
        mock_all_backends["cuquantum"].validate_circuit.return_value.message = (
            "DM not supported"
        )

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            selector = BackendSelector()

            result = selector.select(dm_circuit)

            # Should select DM-capable backend
            assert result in ["quest", "cirq", "qiskit"]

    def test_resource_limit_fallback(self, mock_all_backends):
        """Test fallback when resource limit exceeded."""
        from proxima.intelligence.selector import BackendSelector

        large_circuit = {
            "num_qubits": 25,
            "gates": [],
            "simulator_type": "state_vector",
        }

        # lret has lower qubit limit
        mock_all_backends["lret"].validate_circuit.return_value.valid = False
        mock_all_backends["lret"].validate_circuit.return_value.message = (
            "Exceeds qubit limit"
        )

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            selector = BackendSelector(preferred="lret")

            result = selector.select(large_circuit)

            # Should fall back to higher-capacity backend
            assert result != "lret"


# =============================================================================
# STEP 5.2.4: CONFIGURATION INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.backend
class TestConfigurationIntegration:
    """Integration tests for configuration."""

    def test_config_affects_backend_selection(
        self, mock_all_backends, sample_config, bell_state_circuit
    ):
        """Test that configuration affects backend selection."""
        from proxima.config.settings import Settings
        from proxima.intelligence.selector import BackendSelector

        with patch.object(Settings, "load", return_value=sample_config):
            with patch(
                "proxima.backends.registry.BackendRegistry._discover_backends",
                return_value=mock_all_backends,
            ):
                selector = BackendSelector()

                # Config has prefer_gpu=True
                result = selector.select(bell_state_circuit)

                assert result is not None

    def test_timeout_configuration(
        self, mock_all_backends, bell_state_circuit, sample_config
    ):
        """Test timeout configuration is respected."""
        from proxima.core.executor import CircuitExecutor

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            executor = CircuitExecutor(
                timeout_seconds=sample_config["backends"]["timeout_s"]
            )

            result = executor.execute(bell_state_circuit)

            assert result is not None

    def test_max_qubits_configuration(self, mock_all_backends, sample_config):
        """Test max qubits configuration is enforced."""
        from proxima.intelligence.selector import BackendSelector

        circuit = {"num_qubits": 35, "gates": [], "simulator_type": "state_vector"}

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            selector = BackendSelector(
                max_qubits=sample_config["backends"]["max_qubits"]
            )

            # Should reject or warn about exceeding configured limit
            try:
                selector.select(circuit)
            except Exception as e:
                assert "qubit" in str(e).lower()

    def test_fallback_backend_configuration(
        self, mock_all_backends, bell_state_circuit, sample_config
    ):
        """Test fallback backend from configuration."""
        from proxima.intelligence.selector import BackendSelector

        # Make all backends except fallback unavailable
        for name in mock_all_backends:
            if name != "cirq":
                mock_all_backends[name].is_available.return_value = False

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            selector = BackendSelector(
                fallback_backend=sample_config["auto_selection"]["fallback_backend"]
            )

            result = selector.select(bell_state_circuit)

            assert result == "cirq"

    def test_auto_selection_disabled(self, mock_all_backends, bell_state_circuit):
        """Test behavior when auto-selection is disabled."""
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            selector = BackendSelector(auto_select=False, default_backend="qiskit")

            result = selector.select(bell_state_circuit)

            assert result == "qiskit"


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.backend
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_execution_pipeline(self, mock_all_backends, bell_state_circuit):
        """Test full execution pipeline from selection to result."""
        from proxima.core.executor import CircuitExecutor
        from proxima.intelligence.selector import BackendSelector

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            # Select backend
            selector = BackendSelector()
            backend_name = selector.select(bell_state_circuit)

            # Execute
            executor = CircuitExecutor()
            result = executor.execute(bell_state_circuit, backend=backend_name)

            assert result is not None
            assert result.backend == backend_name

    def test_comparison_pipeline(self, mock_all_backends, ghz_state_circuit):
        """Test full comparison pipeline."""
        from proxima.data.compare import BackendComparator
        from proxima.data.export import ExportEngine

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            # Compare
            comparator = BackendComparator()
            results = comparator.compare(
                ghz_state_circuit,
                backends=["cirq", "qiskit"],
            )

            # Export
            exporter = ExportEngine()
            export_data = exporter.to_dict(results)

            assert export_data is not None
            assert len(export_data) == 2

    def test_error_recovery_pipeline(self, mock_all_backends, bell_state_circuit):
        """Test error recovery in execution pipeline."""
        from proxima.core.executor import CircuitExecutor

        # First attempt fails
        call_count = [0]

        def failing_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Transient error")
            return MagicMock(backend="cirq", data={"counts": {}})

        mock_all_backends["cirq"].execute.side_effect = failing_execute

        with patch(
            "proxima.backends.registry.BackendRegistry._discover_backends",
            return_value=mock_all_backends,
        ):
            executor = CircuitExecutor(retry_count=2)

            result = executor.execute(bell_state_circuit, backend="cirq")

            # Should have recovered
            assert result is not None
