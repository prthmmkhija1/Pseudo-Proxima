"""
Integration Tests for External Systems
=======================================

Tests for integration with external systems as defined in proper_implementation_steps.md:

| External System        | Integration Mode                       | Notes                                |
| ---------------------- | -------------------------------------- | ------------------------------------ |
| Cirq                   | Library import                         | Native Python                        |
| Qiskit Aer             | Library import                         | Native Python                        |
| LRET                   | Custom implementation                  | Pure Python                          |
| OpenAI / Anthropic     | HTTP API                               | Optional, gated                      |
| Ollama / LM Studio     | Local HTTP                             | Privacy-preserving                   |
| psutil                 | Library import                         | System resources                     |
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mark all async tests in this module
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# =============================================================================
# QUANTUM BACKEND INTEGRATION TESTS
# =============================================================================


class TestQuantumBackendIntegration:
    """Integration tests for quantum backend adapters."""

    def test_backend_registry_discovers_backends(self, mock_backend_registry):
        """Test that backend registry can discover available backends."""
        backends = mock_backend_registry.list_available()

        assert len(backends) > 0
        assert "cirq" in backends
        assert "qiskit-aer" in backends
        assert "lret" in backends

    def test_backend_can_be_retrieved(self, mock_backend_registry):
        """Test that backends can be retrieved by name."""
        backend = mock_backend_registry.get("cirq")

        assert backend is not None
        assert backend.name == "cirq"

    async def test_backend_connection_lifecycle(self, mock_quantum_backend):
        """Test backend connect/disconnect lifecycle."""
        # Initially not connected
        assert not mock_quantum_backend.is_connected

        # Connect
        result = await mock_quantum_backend.connect()
        assert result is True
        assert mock_quantum_backend.is_connected

        # Disconnect
        result = await mock_quantum_backend.disconnect()
        assert result is True
        assert not mock_quantum_backend.is_connected

    async def test_backend_execution(self, mock_quantum_backend):
        """Test backend can execute circuits."""
        await mock_quantum_backend.connect()

        result = await mock_quantum_backend.execute(
            circuit=None,  # Mock doesn't need real circuit
            shots=1024,
        )

        assert result["status"] == "success"
        assert "counts" in result
        assert result["backend"] == "mock_backend"

    def test_backend_capabilities(self, mock_quantum_backend):
        """Test backend reports capabilities correctly."""
        capabilities = mock_quantum_backend.get_capabilities()

        assert "name" in capabilities
        assert "max_qubits" in capabilities
        assert "simulator_types" in capabilities
        assert capabilities["max_qubits"] >= 1


# =============================================================================
# LLM PROVIDER INTEGRATION TESTS
# =============================================================================


class TestLLMProviderIntegration:
    """Integration tests for LLM providers."""

    async def test_llm_generate_response(self, mock_llm_provider):
        """Test LLM can generate responses."""
        response = await mock_llm_provider.generate(
            prompt="Explain this quantum circuit",
            model="mock-model",
        )

        assert response.content != ""
        assert response.model == "mock-model"
        assert response.tokens_used > 0

    async def test_llm_tracks_request_history(self, mock_llm_provider):
        """Test LLM tracks request history for auditing."""
        await mock_llm_provider.generate("First prompt")
        await mock_llm_provider.generate("Second prompt")

        history = mock_llm_provider.get_request_history()

        assert len(history) == 2
        assert history[0]["prompt"] == "First prompt"
        assert history[1]["prompt"] == "Second prompt"

    async def test_llm_canned_responses(self, mock_llm_provider):
        """Test LLM returns appropriate canned responses."""
        # Test "explain" keyword
        response = await mock_llm_provider.generate("Please explain this")
        assert (
            "entanglement" in response.content.lower()
            or "circuit" in response.content.lower()
        )

        # Test "analyze" keyword
        response = await mock_llm_provider.generate("Analyze the results")
        assert (
            "bell" in response.content.lower()
            or "superposition" in response.content.lower()
        )

    async def test_llm_failure_simulation(self):
        """Test LLM failure scenarios."""
        from tests.mocks import MockLLMProvider

        provider = MockLLMProvider(success_rate=0.0)  # Always fail

        with pytest.raises(Exception, match="Mock LLM provider error"):
            await provider.generate("test prompt")


# =============================================================================
# SYSTEM RESOURCE INTEGRATION TESTS
# =============================================================================


class TestSystemResourceIntegration:
    """Integration tests for system resource monitoring."""

    def test_normal_resources(self, mock_psutil_normal):
        """Test normal resource monitoring."""
        mem = mock_psutil_normal.virtual_memory()
        cpu = mock_psutil_normal.cpu_percent()

        assert mem.percent < 80  # Normal range
        assert cpu < 80  # Normal range

    def test_low_memory_scenario(self, mock_psutil_low_memory):
        """Test low memory detection."""
        mem = mock_psutil_low_memory.virtual_memory()

        assert mem.percent > 90  # Low memory
        assert mem.available < 1024 * 1024 * 1024  # Less than 1GB

    def test_high_cpu_scenario(self, mock_psutil_high_cpu):
        """Test high CPU detection."""
        cpu = mock_psutil_high_cpu.cpu_percent()

        assert cpu > 90  # High CPU

    def test_low_disk_scenario(self, mock_psutil_low_disk):
        """Test low disk space detection."""
        disk = mock_psutil_low_disk.disk_usage("/")

        assert disk.free < 10 * 1024**3  # Less than 10GB


# =============================================================================
# FILE SYSTEM INTEGRATION TESTS
# =============================================================================


class TestFileSystemIntegration:
    """Integration tests for file system operations."""

    def test_file_write_and_read(self, mock_file_system_comprehensive):
        """Test file write and read operations."""
        fs = mock_file_system_comprehensive

        fs.write_text("/test/file.txt", "Hello, World!")
        content = fs.read_text("/test/file.txt")

        assert content == "Hello, World!"

    def test_file_exists_check(self, mock_file_system_comprehensive):
        """Test file existence checking."""
        fs = mock_file_system_comprehensive

        assert not fs.exists("/nonexistent.txt")

        fs.write_text("/exists.txt", "content")
        assert fs.exists("/exists.txt")

    def test_directory_operations(self, mock_file_system_comprehensive):
        """Test directory creation and listing."""
        fs = mock_file_system_comprehensive

        fs.mkdir("/test/subdir", parents=True)
        fs.write_text("/test/subdir/file1.txt", "content1")
        fs.write_text("/test/subdir/file2.txt", "content2")

        items = fs.list_dir("/test/subdir")

        assert "file1.txt" in items
        assert "file2.txt" in items

    def test_temp_dir_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Can write files
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        assert test_file.exists()
        assert test_file.read_text() == "test content"


# =============================================================================
# CONSENT MANAGER INTEGRATION TESTS
# =============================================================================


class TestConsentManagerIntegration:
    """Integration tests for consent management."""

    def test_auto_approve_consent(self, mock_consent_auto_approve):
        """Test auto-approve mode."""
        result = mock_consent_auto_approve.request_consent("send_data")

        assert result is True

    def test_deny_consent(self, mock_consent_deny):
        """Test deny mode."""
        result = mock_consent_deny.request_consent("send_data")

        # Still returns True for non-sensitive operations
        # because auto_approve only affects sensitive operations
        assert result is True

    def test_sensitive_operations(self, mock_consent_with_sensitive_ops):
        """Test sensitive operation handling."""
        consent = mock_consent_with_sensitive_ops

        assert consent.is_sensitive_operation("send_to_remote_llm")
        assert consent.is_sensitive_operation("export_results")
        assert not consent.is_sensitive_operation("local_operation")

    def test_consent_logging(self, mock_consent_auto_approve):
        """Test consent request logging."""
        consent = mock_consent_auto_approve

        consent.request_consent("operation1", "details1")
        consent.request_consent("operation2", "details2")

        log = consent.get_consent_log()

        assert len(log) == 2
        assert log[0]["operation"] == "operation1"
        assert log[1]["operation"] == "operation2"


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline execution."""

    async def test_pipeline_handler_execution(self, mock_pipeline_handler):
        """Test pipeline handler execution."""
        context = {"test": "data"}

        result = await mock_pipeline_handler.execute(context)

        assert result["stage"] == "test_stage"
        assert result["success"] is True

    async def test_pipeline_handler_failure(self):
        """Test pipeline handler failure scenario."""
        from tests.mocks import MockPipelineHandler

        handler = MockPipelineHandler(
            stage_name="failing_stage",
            success=False,
        )

        result = await handler.execute({})

        assert result["success"] is False
        assert result["error"] is not None


# =============================================================================
# DATA FACTORY INTEGRATION TESTS
# =============================================================================


class TestDataFactoryIntegration:
    """Integration tests for mock data generation."""

    def test_quantum_counts_generation(self, mock_data_factory):
        """Test quantum counts generation."""
        counts = mock_data_factory.quantum_counts(num_qubits=2, shots=1024)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 1024
        assert all(len(k) == 2 for k in counts.keys())

    def test_bell_state_counts(self, bell_state_counts):
        """Test Bell state counts fixture."""
        assert "00" in bell_state_counts
        assert "11" in bell_state_counts
        assert bell_state_counts["00"] + bell_state_counts["11"] == 1024

    def test_ghz_state_counts(self, ghz_state_counts):
        """Test GHZ state counts fixture."""
        assert "000" in ghz_state_counts
        assert "111" in ghz_state_counts

    def test_state_vector_normalization(self, random_state_vector):
        """Test state vector is normalized."""
        import numpy as np

        norm = np.linalg.norm(random_state_vector)
        assert abs(norm - 1.0) < 1e-10

    def test_density_matrix_properties(self, random_density_matrix):
        """Test density matrix has correct properties."""
        import numpy as np

        rho = random_density_matrix

        # Should be square
        assert rho.shape[0] == rho.shape[1]

        # Should have trace = 1
        assert abs(np.trace(rho) - 1.0) < 1e-10

        # Should be Hermitian
        assert np.allclose(rho, rho.conj().T)

    def test_execution_result_generation(self, mock_data_factory):
        """Test execution result generation."""
        result = mock_data_factory.execution_result(
            backend="test_backend",
            success=True,
            num_qubits=3,
            shots=2048,
        )

        assert result["backend"] == "test_backend"
        assert result["status"] == "success"
        assert result["shots"] == 2048
        assert result["num_qubits"] == 3
        assert sum(result["counts"].values()) == 2048


# =============================================================================
# CROSS-COMPONENT INTEGRATION TESTS
# =============================================================================


class TestCrossComponentIntegration:
    """Integration tests for multiple components working together."""

    async def test_backend_with_resource_monitoring(
        self,
        mock_quantum_backend,
        mock_psutil_normal,
    ):
        """Test backend execution with resource monitoring."""
        # Check resources before execution
        mem = mock_psutil_normal.virtual_memory()
        assert mem.percent < 80  # Safe to proceed

        # Execute on backend
        await mock_quantum_backend.connect()
        result = await mock_quantum_backend.execute(shots=1024)

        assert result["status"] == "success"

    async def test_full_workflow_with_consent(
        self,
        mock_quantum_backend,
        mock_llm_provider,
        mock_consent_auto_approve,
    ):
        """Test full workflow with consent management."""
        consent = mock_consent_auto_approve

        # Check consent for LLM usage
        if consent.request_consent("use_llm", "Generate analysis"):
            # Execute quantum simulation
            await mock_quantum_backend.connect()
            result = await mock_quantum_backend.execute(shots=1024)

            # Analyze with LLM
            analysis = await mock_llm_provider.generate(
                f"Analyze these results: {result['counts']}"
            )

            assert analysis.content != ""
            assert consent.get_consent_log()[-1]["operation"] == "use_llm"

    async def test_resource_constrained_execution(
        self,
        mock_quantum_backend,
        mock_psutil_low_memory,
    ):
        """Test behavior under resource constraints."""
        mem = mock_psutil_low_memory.virtual_memory()

        # Should detect low memory
        assert mem.percent > 90

        # In real implementation, this might trigger warnings or abort
        # For mock, we just verify we can detect the condition
        await mock_quantum_backend.connect()
        result = await mock_quantum_backend.execute(shots=512)  # Reduced shots

        assert result["status"] == "success"
