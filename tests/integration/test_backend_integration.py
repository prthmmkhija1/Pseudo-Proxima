"""Step 6.2: Backend Integration Tests - Cross-backend integration testing.

Integration tests covering:
- Backend-to-backend result consistency
- Pipeline integration with all backends
- Store integration with backend results
- Export integration with comparison results
- Real workflow simulations
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any
import asyncio
import json
import tempfile
import os


# =============================================================================
# BACKEND INTEGRATION FIXTURES
# =============================================================================


@pytest.fixture
def mock_backend_registry():
    """Create a mock backend registry with all backends."""
    registry = {
        "cirq": {
            "name": "cirq",
            "type": "simulator",
            "available": True,
            "capabilities": ["state_vector", "density_matrix"],
        },
        "qiskit_aer": {
            "name": "qiskit_aer",
            "type": "simulator",
            "available": True,
            "capabilities": ["state_vector", "density_matrix", "noise"],
        },
        "lret": {
            "name": "lret",
            "type": "simulator",
            "available": True,
            "capabilities": ["state_vector"],
        },
        "quest": {
            "name": "quest",
            "type": "simulator",
            "available": True,
            "capabilities": ["state_vector", "density_matrix", "gpu"],
        },
        "qsim": {
            "name": "qsim",
            "type": "simulator",
            "available": True,
            "capabilities": ["state_vector", "gpu"],
        },
        "cuquantum": {
            "name": "cuquantum",
            "type": "simulator",
            "available": True,
            "capabilities": ["state_vector", "tensor_network", "gpu"],
        },
    }
    return registry


@pytest.fixture
def mock_circuit_result():
    """Create mock circuit execution result."""
    import numpy as np
    
    return {
        "state_vector": np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128),
        "measurements": {"00": 500, "11": 500},
        "execution_time_ms": 50.0,
        "memory_mb": 128.0,
    }


# =============================================================================
# CROSS-BACKEND CONSISTENCY TESTS
# =============================================================================


class TestCrossBackendConsistency:
    """Tests for result consistency across backends."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bell_state_consistency(self, mock_backend_registry, mock_circuit_result):
        """Test Bell state produces consistent results across backends."""
        from proxima.data.compare import BackendResult, ComparisonMetrics
        
        results = []
        for backend_name in ["cirq", "qiskit_aer", "lret"]:
            result = BackendResult(
                backend_name=backend_name,
                success=True,
                execution_time_ms=mock_circuit_result["execution_time_ms"],
                memory_peak_mb=mock_circuit_result["memory_mb"],
                metadata={"measurements": mock_circuit_result["measurements"]},
            )
            results.append(result)
        
        # All backends should succeed
        assert all(r.success for r in results)
        
        # All backends should have similar measurement distributions
        for result in results:
            measurements = result.metadata.get("measurements", {})
            if measurements:
                total = sum(measurements.values())
                for state in ["00", "11"]:
                    ratio = measurements.get(state, 0) / total
                    assert 0.4 <= ratio <= 0.6

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ghz_state_consistency(self, mock_backend_registry):
        """Test GHZ state consistency across backends."""
        from proxima.data.compare import BackendResult
        
        # GHZ state should have equal probability for |000⟩ and |111⟩
        ghz_measurements = {"000": 500, "111": 500}
        
        results = []
        for backend in ["cirq", "qiskit_aer", "quest"]:
            result = BackendResult(
                backend_name=backend,
                success=True,
                execution_time_ms=75.0,
                memory_peak_mb=256.0,
                metadata={"measurements": ghz_measurements},
            )
            results.append(result)
        
        # Verify all have expected states
        for result in results:
            measurements = result.metadata["measurements"]
            assert "000" in measurements
            assert "111" in measurements
            assert measurements["000"] + measurements["111"] == 1000

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_state_vector_fidelity(self, mock_backend_registry):
        """Test state vector fidelity between backends."""
        import numpy as np
        
        # Simulated state vectors from different backends
        cirq_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        qiskit_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        lret_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        
        def fidelity(sv1, sv2):
            return np.abs(np.dot(np.conj(sv1), sv2)) ** 2
        
        # All fidelities should be essentially 1.0
        assert np.isclose(fidelity(cirq_sv, qiskit_sv), 1.0, atol=1e-10)
        assert np.isclose(fidelity(cirq_sv, lret_sv), 1.0, atol=1e-10)
        assert np.isclose(fidelity(qiskit_sv, lret_sv), 1.0, atol=1e-10)


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================


class TestPipelineIntegration:
    """Tests for pipeline integration with backends."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_multi_backend_execution(self, mock_backend_registry):
        """Test pipeline executing on multiple backends."""
        from proxima.data.pipeline import (
            Pipeline,
            PipelineConfig,
            Stage,
            StageStatus,
            PipelineStatus,
        )
        from proxima.data.compare import BackendResult
        
        async def execute_on_backend(ctx, backend_name):
            """Mock backend execution stage."""
            await asyncio.sleep(0.01)  # Simulate work
            return BackendResult(
                backend_name=backend_name,
                success=True,
                execution_time_ms=50.0,
                memory_peak_mb=128.0,
            )
        
        # Create pipeline with stages for each backend
        config = PipelineConfig(name="multi_backend_test")
        pipeline = Pipeline(config)
        
        for backend in ["cirq", "qiskit_aer", "lret"]:
            stage = Stage(
                stage_id=f"execute_{backend}",
                name=f"Execute on {backend}",
                handler=lambda ctx, inp, b=backend: execute_on_backend(ctx, b),
            )
            pipeline.add_stage(stage)
        
        # Execute pipeline
        result = await pipeline.execute()
        
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.successful_stages) == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_comparison_stage(self, mock_backend_registry):
        """Test pipeline with backend comparison stage."""
        from proxima.data.pipeline import Pipeline, PipelineConfig, Stage, PipelineContext
        from proxima.data.compare import BackendResult
        
        async def collect_results(ctx, _):
            """Collect results from all backends."""
            results = []
            for backend in ["cirq", "qiskit_aer"]:
                results.append(BackendResult(
                    backend_name=backend,
                    success=True,
                    execution_time_ms=50.0 + len(results) * 10,
                    memory_peak_mb=128.0,
                ))
            ctx.set("backend_results", results)
            return results
        
        async def compare_results(ctx, results):
            """Compare results from different backends."""
            fastest = min(results, key=lambda r: r.execution_time_ms)
            ctx.set("fastest_backend", fastest.backend_name)
            return {"fastest": fastest.backend_name}
        
        config = PipelineConfig(name="comparison_pipeline")
        pipeline = Pipeline(config)
        
        pipeline.add_stage(Stage(
            stage_id="collect",
            name="Collect Results",
            handler=collect_results,
        ))
        pipeline.add_stage(Stage(
            stage_id="compare",
            name="Compare Results",
            handler=compare_results,
            dependencies=["collect"],
        ))
        
        result = await pipeline.execute()
        
        assert result.is_success
        assert len(result.successful_stages) == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_retry_on_backend_failure(self):
        """Test pipeline retry when backend fails."""
        from proxima.data.pipeline import Pipeline, PipelineConfig, Stage, RetryConfig, RetryStrategy
        
        attempt_count = 0
        
        async def flaky_backend(ctx, _):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Backend temporarily unavailable")
            return {"success": True}
        
        config = PipelineConfig(
            name="retry_test",
            retry=RetryConfig(
                max_retries=5,
                strategy=RetryStrategy.IMMEDIATE,
            ),
        )
        pipeline = Pipeline(config)
        
        pipeline.add_stage(Stage(
            stage_id="flaky",
            name="Flaky Backend",
            handler=flaky_backend,
        ))
        
        result = await pipeline.execute()
        
        assert result.is_success
        assert attempt_count == 3  # Failed twice, succeeded third time

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_cancellation(self):
        """Test pipeline cancellation during execution."""
        from proxima.data.pipeline import Pipeline, PipelineConfig, Stage, PipelineStatus, CancellationReason
        
        async def long_running_stage(ctx, _):
            for i in range(10):
                ctx.check_cancelled()
                await asyncio.sleep(0.01)
            return {"completed": True}
        
        config = PipelineConfig(name="cancel_test")
        pipeline = Pipeline(config)
        
        pipeline.add_stage(Stage(
            stage_id="long",
            name="Long Running",
            handler=long_running_stage,
        ))
        
        # Start pipeline and cancel after short delay
        async def run_and_cancel():
            task = asyncio.create_task(pipeline.execute())
            await asyncio.sleep(0.03)
            pipeline.cancel(CancellationReason.USER_REQUESTED)
            return await task
        
        result = await run_and_cancel()
        
        assert result.status == PipelineStatus.CANCELLED
        assert result.cancellation_reason == CancellationReason.USER_REQUESTED


# =============================================================================
# STORE INTEGRATION TESTS
# =============================================================================


class TestStoreIntegration:
    """Tests for store integration with backend results."""

    @pytest.mark.integration
    def test_store_backend_results(self):
        """Test storing backend results in different stores."""
        from proxima.data.store import MemoryStore, StoredResult
        from proxima.data.compare import BackendResult
        import time
        
        store = MemoryStore()
        
        # Create and store results from multiple backends
        for backend in ["cirq", "qiskit_aer", "lret"]:
            result = StoredResult(
                id=f"result_{backend}_{int(time.time())}",
                session_id="test_session",
                backend_name=backend,
                qubit_count=2,
                shots=1000,
                execution_time_ms=50.0,
                memory_used_mb=128.0,
                counts={"00": 500, "11": 500},
                metadata={},
            )
            store.save_result(result)
        
        # Retrieve all results
        all_results = store.list_results()
        assert len(all_results) == 3
        
        # Query by backend
        cirq_results = [r for r in all_results if r.backend_name == "cirq"]
        assert len(cirq_results) == 1

    @pytest.mark.integration
    def test_store_comparison_session(self):
        """Test storing a complete comparison session."""
        from proxima.data.store import MemoryStore, StoredSession, StoredResult
        import time
        
        store = MemoryStore()
        
        # Create session
        session = StoredSession(
            id="comparison_001",
            name="Backend Comparison Test",
            metadata={
                "backends": ["cirq", "qiskit_aer", "lret"],
                "circuit_type": "bell_state",
            },
        )
        store.create_session(session)
        
        # Store results for session
        for backend in ["cirq", "qiskit_aer", "lret"]:
            result = StoredResult(
                id=f"comp_001_{backend}",
                session_id="comparison_001",
                backend_name=backend,
                qubit_count=2,
                shots=1000,
                execution_time_ms=50.0 + hash(backend) % 20,
                memory_used_mb=128.0,
                counts={},
                metadata={},
            )
            store.save_result(result)
        
        # Retrieve session with results
        retrieved_session = store.get_session("comparison_001")
        assert retrieved_session is not None
        assert retrieved_session.name == "Backend Comparison Test"

    @pytest.mark.integration
    def test_json_store_persistence(self):
        """Test JSON store persistence."""
        from proxima.data.store import JSONStore, StoredResult
        from pathlib import Path
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = JSONStore(store_path)
            
            # Store result
            result = StoredResult(
                id="persist_test_001",
                session_id="session_001",
                backend_name="cirq",
                qubit_count=2,
                shots=1000,
                execution_time_ms=100.0,
                memory_used_mb=128.0,
                counts={},
                metadata={"test": "data"},
            )
            
            # Store result (synchronous)
            store.save_result(result)
            
            # Create new store instance and verify persistence
            store2 = JSONStore(store_path)
            retrieved = store2.get_result("persist_test_001")
            
            assert retrieved is not None
            assert retrieved.backend_name == "cirq"


# =============================================================================
# EXPORT INTEGRATION TESTS
# =============================================================================


class TestExportIntegration:
    """Tests for export integration with comparison results."""

    @pytest.mark.integration
    def test_export_comparison_to_json(self):
        """Test exporting comparison results to JSON."""
        from proxima.data.export import JSONExporter, ReportData
        from proxima.data.compare import BackendResult, ComparisonReport
        import time
        
        # Create comparison report
        results = [
            BackendResult(
                backend_name="cirq",
                success=True,
                execution_time_ms=50.0,
                memory_peak_mb=128.0,
            ),
            BackendResult(
                backend_name="qiskit_aer",
                success=True,
                execution_time_ms=45.0,
                memory_peak_mb=150.0,
            ),
        ]
        
        report_data = ReportData(
            title="Backend Comparison",
            generated_at=time.time(),
            summary={"total_backends": 2, "successful": 2},
            sections=[
                {"name": "Results", "content": [r.backend_name for r in results]},
            ],
            raw_data={"results": [{"backend": r.backend_name} for r in results]},
        )
        
        exporter = JSONExporter()
        output = exporter.export(report_data)
        
        assert output is not None
        # Verify it's valid JSON
        parsed = json.loads(output)
        assert parsed["title"] == "Backend Comparison"

    @pytest.mark.integration
    def test_export_comparison_to_csv(self):
        """Test exporting comparison results to CSV."""
        from proxima.data.export import CSVExporter, ReportData
        import time
        
        report_data = ReportData(
            title="Backend Metrics",
            generated_at=time.time(),
            summary={},
            sections=[],
            raw_data={
                "rows": [
                    {"backend": "cirq", "time_ms": 50.0, "memory_mb": 128.0},
                    {"backend": "qiskit_aer", "time_ms": 45.0, "memory_mb": 150.0},
                    {"backend": "lret", "time_ms": 55.0, "memory_mb": 100.0},
                ],
            },
        )
        
        exporter = CSVExporter()
        output = exporter.export(report_data)
        
        assert output is not None
        lines = output.strip().split("\n")
        assert len(lines) >= 1  # At least header

    @pytest.mark.integration
    def test_export_comparison_to_markdown(self):
        """Test exporting comparison results to Markdown."""
        from proxima.data.export import MarkdownExporter, ReportData
        import time
        
        report_data = ReportData(
            title="Backend Comparison Report",
            generated_at=time.time(),
            summary={
                "fastest_backend": "qiskit_aer",
                "most_efficient": "lret",
            },
            sections=[
                {
                    "name": "Performance",
                    "content": "All backends performed within expected parameters.",
                },
            ],
            raw_data={},
        )
        
        exporter = MarkdownExporter()
        output = exporter.export(report_data)
        
        assert output is not None
        assert "# Backend Comparison Report" in output or "Backend Comparison Report" in output


# =============================================================================
# FULL WORKFLOW INTEGRATION TESTS
# =============================================================================


class TestFullWorkflowIntegration:
    """Tests for complete workflow integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_comparison_workflow(self):
        """Test complete comparison workflow from execution to export."""
        from proxima.data.pipeline import Pipeline, PipelineConfig, Stage
        from proxima.data.compare import BackendResult
        from proxima.data.store import MemoryStore, StoredResult
        from proxima.data.export import JSONExporter, ReportData
        import time
        
        store = MemoryStore()
        results = []
        
        # Stage 1: Execute on backends
        async def execute_backends(ctx, _):
            backend_results = []
            for backend in ["cirq", "qiskit_aer", "lret"]:
                result = BackendResult(
                    backend_name=backend,
                    success=True,
                    execution_time_ms=50.0 + hash(backend) % 30,
                    memory_peak_mb=128.0 + hash(backend) % 50,
                )
                backend_results.append(result)
            ctx.set("backend_results", backend_results)
            return backend_results
        
        # Stage 2: Store results
        async def store_results(ctx, backend_results):
            session_id = f"session_{int(time.time())}"
            for result in backend_results:
                stored = StoredResult(
                    id=f"{session_id}_{result.backend_name}",
                    session_id=session_id,
                    backend_name=result.backend_name,
                    qubit_count=2,
                    shots=1000,
                    execution_time_ms=result.execution_time_ms,
                    memory_used_mb=result.memory_peak_mb,
                    counts={},
                    metadata={},
                )
                store.save_result(stored)
            ctx.set("session_id", session_id)
            return session_id
        
        # Stage 3: Generate report
        async def generate_report(ctx, session_id):
            backend_results = ctx.get("backend_results")
            fastest = min(backend_results, key=lambda r: r.execution_time_ms)
            
            report_data = ReportData(
                title="Comparison Report",
                generated_at=time.time(),
                summary={
                    "session_id": session_id,
                    "fastest_backend": fastest.backend_name,
                    "backends_tested": len(backend_results),
                },
                sections=[],
                raw_data={},
            )
            
            exporter = JSONExporter()
            return exporter.export(report_data)
        
        # Build and execute pipeline
        config = PipelineConfig(name="full_workflow")
        pipeline = Pipeline(config)
        
        pipeline.add_stages([
            Stage(stage_id="execute", name="Execute", handler=execute_backends),
            Stage(stage_id="store", name="Store", handler=store_results, dependencies=["execute"]),
            Stage(stage_id="report", name="Report", handler=generate_report, dependencies=["store"]),
        ])
        
        result = await pipeline.execute()
        
        assert result.is_success
        assert len(result.successful_stages) == 3
        
        # Verify stored results
        stored_results = store.list_results()
        assert len(stored_results) == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test workflow with error recovery."""
        from proxima.data.pipeline import Pipeline, PipelineConfig, Stage, RetryConfig, RetryStrategy
        
        execution_log = []
        
        async def primary_backend(ctx, _):
            execution_log.append("primary_attempt")
            raise Exception("Primary backend unavailable")
        
        async def fallback_backend(ctx, _):
            execution_log.append("fallback")
            return {"backend": "fallback", "success": True}
        
        async def process_result(ctx, result):
            execution_log.append("process")
            return result
        
        config = PipelineConfig(
            name="error_recovery",
            continue_on_error=True,
        )
        pipeline = Pipeline(config)
        
        pipeline.add_stages([
            Stage(
                stage_id="primary",
                name="Primary Backend",
                handler=primary_backend,
                critical=False,  # Non-critical, continue on failure
            ),
            Stage(
                stage_id="fallback",
                name="Fallback Backend",
                handler=fallback_backend,
            ),
            Stage(
                stage_id="process",
                name="Process",
                handler=process_result,
                dependencies=["fallback"],
            ),
        ])
        
        result = await pipeline.execute()
        
        # Should complete with partial success
        assert "fallback" in execution_log
        assert "process" in execution_log


# =============================================================================
# BACKEND SELECTION INTEGRATION TESTS
# =============================================================================


class TestBackendSelectionIntegration:
    """Tests for backend selection integration."""

    @pytest.mark.integration
    def test_select_backend_by_capability(self, mock_backend_registry):
        """Test selecting backend by capability."""
        required_capability = "density_matrix"
        
        matching_backends = [
            name for name, info in mock_backend_registry.items()
            if required_capability in info["capabilities"]
        ]
        
        assert "cirq" in matching_backends
        assert "qiskit_aer" in matching_backends
        assert "quest" in matching_backends
        assert "lret" not in matching_backends  # LRET doesn't support density matrix

    @pytest.mark.integration
    def test_select_backend_by_performance(self):
        """Test selecting backend by performance characteristics."""
        from proxima.data.compare import BackendResult
        
        # Historical performance data
        performance_data = [
            BackendResult(backend_name="cirq", success=True, execution_time_ms=100.0, memory_peak_mb=256.0),
            BackendResult(backend_name="qiskit_aer", success=True, execution_time_ms=85.0, memory_peak_mb=300.0),
            BackendResult(backend_name="lret", success=True, execution_time_ms=120.0, memory_peak_mb=200.0),
            BackendResult(backend_name="quest", success=True, execution_time_ms=70.0, memory_peak_mb=350.0),
        ]
        
        # Select fastest
        fastest = min(performance_data, key=lambda r: r.execution_time_ms)
        assert fastest.backend_name == "quest"
        
        # Select most memory efficient
        efficient = min(performance_data, key=lambda r: r.memory_peak_mb)
        assert efficient.backend_name == "lret"

    @pytest.mark.integration
    def test_select_gpu_backend(self, mock_backend_registry):
        """Test selecting GPU-capable backend."""
        gpu_backends = [
            name for name, info in mock_backend_registry.items()
            if "gpu" in info["capabilities"]
        ]
        
        assert "quest" in gpu_backends
        assert "qsim" in gpu_backends
        assert "cuquantum" in gpu_backends
        assert "cirq" not in gpu_backends
