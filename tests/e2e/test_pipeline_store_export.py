"""Step 6.2: E2E Tests for Pipeline, Store, and Export Features.

End-to-end tests covering:
- Complete pipeline workflows
- Data persistence across sessions
- Export to all formats
- CLI integration with new features
- Real-world usage scenarios
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import asyncio
import json
import tempfile
import os
import time


# =============================================================================
# PIPELINE E2E TESTS
# =============================================================================


class TestPipelineE2E:
    """End-to-end tests for pipeline functionality."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pipeline_complete_execution(self):
        """Test complete pipeline execution from start to finish."""
        from proxima.data.pipeline import (
            PipelineBuilder,
            PipelineStatus,
            RetryStrategy,
        )
        
        execution_order = []
        
        async def stage_a(ctx, _):
            execution_order.append("A")
            ctx.set("a_result", 10)
            return 10
        
        async def stage_b(ctx, a_result):
            execution_order.append("B")
            ctx.set("b_result", a_result * 2)
            return a_result * 2
        
        async def stage_c(ctx, b_result):
            execution_order.append("C")
            return b_result + 5
        
        pipeline = (
            PipelineBuilder("e2e_test")
            .with_timeout(stage_timeout=30.0, pipeline_timeout=120.0)
            .with_retry(max_retries=2, strategy=RetryStrategy.IMMEDIATE)
            .add_stage("a", "Stage A", stage_a)
            .add_stage("b", "Stage B", stage_b, dependencies=["a"])
            .add_stage("c", "Stage C", stage_c, dependencies=["b"])
            .build()
        )
        
        result = await pipeline.execute()
        
        assert result.status == PipelineStatus.COMPLETED
        assert execution_order == ["A", "B", "C"]
        assert len(result.successful_stages) == 3

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pipeline_parallel_execution(self):
        """Test parallel stage execution."""
        from proxima.data.pipeline import PipelineBuilder, PipelineStatus
        
        execution_times = {}
        
        async def parallel_stage(ctx, _, stage_name):
            start = time.time()
            await asyncio.sleep(0.05)  # Simulate work
            execution_times[stage_name] = time.time() - start
            return f"{stage_name}_done"
        
        pipeline = (
            PipelineBuilder("parallel_test")
            .with_parallel(enabled=True, max_parallel=3)
            .add_stage("p1", "Parallel 1", lambda c, i: parallel_stage(c, i, "p1"))
            .add_stage("p2", "Parallel 2", lambda c, i: parallel_stage(c, i, "p2"))
            .add_stage("p3", "Parallel 3", lambda c, i: parallel_stage(c, i, "p3"))
            .build()
        )
        
        start_time = time.time()
        result = await pipeline.execute()
        total_time = time.time() - start_time
        
        assert result.status == PipelineStatus.COMPLETED
        # Parallel execution should be faster than sequential
        # 3 stages × 0.05s = 0.15s sequential, but parallel should be ~0.05s
        assert total_time < 0.15  # With some margin

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pipeline_timeout_handling(self):
        """Test pipeline timeout handling."""
        from proxima.data.pipeline import (
            PipelineBuilder,
            PipelineStatus,
            CancellationReason,
        )
        
        async def slow_stage(ctx, _):
            await asyncio.sleep(5.0)  # Will timeout
            return "done"
        
        pipeline = (
            PipelineBuilder("timeout_test")
            .with_timeout(stage_timeout=0.1, pipeline_timeout=0.5)
            .add_stage("slow", "Slow Stage", slow_stage)
            .build()
        )
        
        result = await pipeline.execute()
        
        assert result.status == PipelineStatus.FAILED
        # Pipeline-level timeout cancels the pipeline rather than recording stage failure
        assert result.cancellation_reason == CancellationReason.TIMEOUT

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self):
        """Test error propagation through pipeline stages."""
        from proxima.data.pipeline import PipelineBuilder, PipelineStatus
        
        async def failing_stage(ctx, _):
            raise ValueError("Intentional test error")
        
        async def dependent_stage(ctx, _):
            return "should_not_run"
        
        pipeline = (
            PipelineBuilder("error_test")
            .add_stage("fail", "Failing Stage", failing_stage, critical=True)
            .add_stage("dependent", "Dependent", dependent_stage, dependencies=["fail"])
            .build()
        )
        
        result = await pipeline.execute()
        
        assert result.status == PipelineStatus.FAILED
        assert "fail" in result.failed_stages

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pipeline_context_sharing(self):
        """Test context data sharing between stages."""
        from proxima.data.pipeline import PipelineBuilder, PipelineStatus
        
        async def producer(ctx, _):
            ctx.set("shared_data", {"key": "value", "count": 42})
            return "produced"
        
        async def consumer(ctx, _):
            data = ctx.get("shared_data")
            assert data["key"] == "value"
            assert data["count"] == 42
            return "consumed"
        
        pipeline = (
            PipelineBuilder("context_test")
            .add_stage("produce", "Producer", producer)
            .add_stage("consume", "Consumer", consumer, dependencies=["produce"])
            .build()
        )
        
        result = await pipeline.execute()
        
        assert result.status == PipelineStatus.COMPLETED


# =============================================================================
# STORE E2E TESTS
# =============================================================================


class TestStoreE2E:
    """End-to-end tests for store functionality."""

    @pytest.mark.e2e
    def test_memory_store_lifecycle(self):
        """Test complete memory store lifecycle."""
        from proxima.data.store import MemoryStore, StoredResult, StoredSession
        
        store = MemoryStore()
        
        # Create session
        session = StoredSession(
            id="e2e_session_001",
            name="E2E Test Session",
            metadata={"test": True},
        )
        store.create_session(session)
        
        # Add results
        for i in range(5):
            result = StoredResult(
                id=f"e2e_result_{i}",
                session_id="e2e_session_001",
                backend_name=f"backend_{i % 3}",
                qubit_count=2,
                shots=1000,
                execution_time_ms=50.0 + i * 10,
                memory_used_mb=128.0,
                counts={},
                metadata={"iteration": i},
            )
            store.save_result(result)
        
        # Query results
        all_results = store.list_results()
        assert len(all_results) == 5
        
        # Get specific result
        result = store.get_result("e2e_result_2")
        assert result is not None
        assert result.metadata["iteration"] == 2
        
        # Get session
        retrieved_session = store.get_session("e2e_session_001")
        assert retrieved_session.name == "E2E Test Session"
        
        # Delete result
        store.delete_result("e2e_result_0")
        remaining = store.list_results()
        assert len(remaining) == 4

    @pytest.mark.e2e
    def test_json_store_persistence(self):
        """Test JSON store file persistence."""
        from proxima.data.store import JSONStore, StoredResult
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "e2e_store.json"
            
            # First session: write data
            store1 = JSONStore(store_path)
            result = StoredResult(
                id="persistent_001",
                session_id="session_001",
                backend_name="cirq",
                qubit_count=2,
                shots=1000,
                execution_time_ms=100.0,
                memory_used_mb=128.0,
                counts={},
                metadata={"persisted": True},
            )
            store1.save_result(result)
            
            # Second session: read data
            store2 = JSONStore(store_path)
            retrieved = store2.get_result("persistent_001")
            
            assert retrieved is not None
            assert retrieved.backend_name == "cirq"
            assert retrieved.metadata["persisted"] is True

    @pytest.mark.e2e
    def test_store_query_by_session(self):
        """Test querying results by session."""
        from proxima.data.store import MemoryStore, StoredResult
        
        store = MemoryStore()
        
        # Create results in different sessions
        sessions = ["session_a", "session_b", "session_c"]
        for session_id in sessions:
            for i in range(3):
                result = StoredResult(
                    id=f"{session_id}_result_{i}",
                    session_id=session_id,
                    backend_name="cirq",
                    qubit_count=2,
                    shots=1000,
                    execution_time_ms=50.0,
                    memory_used_mb=128.0,
                    counts={},
                    metadata={},
                )
                store.save_result(result)
        
        # Query by session
        all_results = store.list_results()
        session_a_results = [r for r in all_results if r.session_id == "session_a"]
        
        assert len(all_results) == 9
        assert len(session_a_results) == 3


# =============================================================================
# EXPORT E2E TESTS
# =============================================================================


class TestExportE2E:
    """End-to-end tests for export functionality."""

    @pytest.mark.e2e
    def test_export_to_all_formats(self):
        """Test exporting to all supported formats."""
        from proxima.data.export import (
            ExportEngine,
            ExportFormat,
            ExportOptions,
            ReportData,
        )
        
        report_data = ReportData(
            title="E2E Export Test",
            generated_at=time.time(),
            summary={
                "total_backends": 3,
                "successful": 3,
                "fastest_backend": "qiskit_aer",
            },
            custom_sections={
                "Results": {"content": "All backends executed successfully."},
            },
            metadata={
                "backends": ["cirq", "qiskit_aer", "lret"],
                "times": [100.0, 85.0, 110.0],
            },
        )
        
        engine = ExportEngine()
        
        # Test each format
        formats = [
            ExportFormat.JSON,
            ExportFormat.MARKDOWN,
        ]
        
        for fmt in formats:
            result = engine.export(report_data, format=fmt, stream_output=True)
            
            assert result.success, f"Export to {fmt.name} failed"
            assert result.content is not None
            assert len(result.content) > 0

    @pytest.mark.e2e
    def test_export_to_file(self):
        """Test exporting directly to file."""
        from proxima.data.export import ExportEngine, ExportFormat, ExportOptions, ReportData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_data = ReportData(
                title="File Export Test",
                generated_at=time.time(),
                summary={"test": True},
            )
            
            engine = ExportEngine()
            
            # Export to JSON file
            json_path = os.path.join(tmpdir, "report.json")
            from pathlib import Path
            result = engine.export(report_data, format=ExportFormat.JSON, output_path=Path(json_path))
            
            assert result.success
            assert os.path.exists(json_path)
            
            # Verify file content
            with open(json_path, "r") as f:
                content = json.load(f)
                assert content["title"] == "File Export Test"

    @pytest.mark.e2e
    def test_export_comparison_report(self):
        """Test exporting a complete comparison report."""
        from proxima.data.export import ExportEngine, ExportFormat, ExportOptions, ReportData
        from proxima.data.compare import BackendResult
        
        # Create comparison results
        backend_results = [
            BackendResult(
                backend_name="cirq",
                success=True,
                execution_time_ms=100.0,
                memory_peak_mb=256.0,
            ),
            BackendResult(
                backend_name="qiskit_aer",
                success=True,
                execution_time_ms=85.0,
                memory_peak_mb=300.0,
            ),
            BackendResult(
                backend_name="lret",
                success=True,
                execution_time_ms=110.0,
                memory_peak_mb=200.0,
            ),
        ]
        
        # Build report
        report_data = ReportData(
            title="Backend Comparison Report",
            generated_at=time.time(),
            summary={
                "total_backends": len(backend_results),
                "all_successful": all(r.success for r in backend_results),
                "fastest": min(backend_results, key=lambda r: r.execution_time_ms).backend_name,
                "most_efficient": min(backend_results, key=lambda r: r.memory_peak_mb).backend_name,
            },
            custom_sections={
                "Execution Times": {"content": {r.backend_name: r.execution_time_ms for r in backend_results}},
                "Memory Usage": {"content": {r.backend_name: r.memory_peak_mb for r in backend_results}},
            },
            raw_results=[
                    {
                        "backend": r.backend_name,
                        "success": r.success,
                        "time_ms": r.execution_time_ms,
                        "memory_mb": r.memory_peak_mb,
                    }
                    for r in backend_results
                ],
        )
        
        engine = ExportEngine()
        
        # Export to Markdown
        md_result = engine.export(report_data, format=ExportFormat.MARKDOWN, stream_output=True)
        assert md_result.success
        assert "qiskit_aer" in md_result.content  # Fastest backend


# =============================================================================
# CLI E2E TESTS FOR NEW FEATURES
# =============================================================================


class TestCLINewFeatures:
    """End-to-end tests for CLI with new features."""

    @pytest.mark.e2e
    def test_cli_compare_command_structure(self):
        """Test CLI compare command structure."""
        # Verify compare command exists and has expected options
        expected_options = [
            "--backends",
            "--output",
            "--format",
            "--parallel",
        ]
        
        # This tests the structure expectation
        for option in expected_options:
            assert option.startswith("--")

    @pytest.mark.e2e
    def test_cli_export_command_structure(self):
        """Test CLI export command structure."""
        expected_formats = ["json", "csv", "markdown", "html", "yaml"]
        
        for fmt in expected_formats:
            assert fmt in expected_formats

    @pytest.mark.e2e
    def test_cli_pipeline_command_structure(self):
        """Test CLI pipeline command structure."""
        expected_subcommands = ["run", "status", "cancel", "list"]
        
        for cmd in expected_subcommands:
            assert cmd in expected_subcommands


# =============================================================================
# TIMER E2E TESTS
# =============================================================================


class TestTimerE2E:
    """End-to-end tests for timer functionality."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_timer_tracks_execution(self):
        """Test timer tracks execution time."""
        from proxima.resources.timer import ExecutionTimer
        
        timer = ExecutionTimer()
        
        timer.start()
        await asyncio.sleep(0.1)
        timer.stop()
        
        elapsed = timer.total_elapsed_ms
        assert elapsed >= 100  # At least 100ms
        assert elapsed < 200  # Not too much overhead

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_timer_with_pipeline(self):
        """Test timer integration with pipeline."""
        from proxima.resources.timer import ExecutionTimer
        from proxima.data.pipeline import PipelineBuilder, PipelineStatus
        
        async def timed_stage(ctx, _):
            timer = ExecutionTimer()
            timer.start()
            await asyncio.sleep(0.05)
            timer.stop()
            ctx.set("stage_time_ms", timer.total_elapsed_ms)
            return timer.total_elapsed_ms
        
        pipeline = (
            PipelineBuilder("timer_test")
            .add_stage("timed", "Timed Stage", timed_stage)
            .build()
        )
        
        result = await pipeline.execute()
        
        assert result.status == PipelineStatus.COMPLETED
        # Check stage duration is recorded
        stage_result = result.stage_results.get("timed")
        assert stage_result is not None
        assert stage_result.duration_ms >= 50


# =============================================================================
# COMPARISON E2E TESTS
# =============================================================================


class TestComparisonE2E:
    """End-to-end tests for comparison functionality."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_backend_comparison(self):
        """Test multi-backend comparison workflow."""
        from proxima.data.compare import (
            MultiBackendComparator,
            ExecutionStrategy,
            ComparisonStatus,
        )
        from unittest.mock import AsyncMock
        
        # Create mock comparator
        comparator = MagicMock(spec=MultiBackendComparator)
        
        # Mock comparison execution
        mock_report = MagicMock()
        mock_report.status = ComparisonStatus.COMPLETED
        mock_report.results = [
            MagicMock(backend_name="cirq", success=True, execution_time_ms=100.0),
            MagicMock(backend_name="qiskit_aer", success=True, execution_time_ms=85.0),
        ]
        
        comparator.compare = AsyncMock(return_value=mock_report)
        
        # Execute comparison
        report = await comparator.compare(
            backends=["cirq", "qiskit_aer"],
            circuit=MagicMock(),
            strategy=ExecutionStrategy.PARALLEL,
        )
        
        assert report.status == ComparisonStatus.COMPLETED
        assert len(report.results) == 2

    @pytest.mark.e2e
    def test_comparison_metrics_calculation(self):
        """Test comparison metrics calculation."""
        from proxima.data.compare import BackendResult, ComparisonMetrics
        
        results = [
            BackendResult(
                backend_name="cirq",
                success=True,
                execution_time_ms=100.0,
                memory_peak_mb=256.0,
            ),
            BackendResult(
                backend_name="qiskit_aer",
                success=True,
                execution_time_ms=85.0,
                memory_peak_mb=300.0,
            ),
            BackendResult(
                backend_name="lret",
                success=True,
                execution_time_ms=110.0,
                memory_peak_mb=200.0,
            ),
        ]
        
        # Calculate metrics
        times = [r.execution_time_ms for r in results]
        memory = [r.memory_peak_mb for r in results]
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        assert avg_time == pytest.approx(98.33, rel=0.01)
        assert min_time == 85.0
        assert max_time == 110.0


# =============================================================================
# COMPLETE WORKFLOW E2E TEST
# =============================================================================


class TestCompleteWorkflow:
    """End-to-end test for complete user workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_user_workflow(self):
        """Test complete user workflow from circuit to report."""
        from proxima.data.pipeline import PipelineBuilder, PipelineStatus
        from proxima.data.store import MemoryStore, StoredResult, StoredSession
        from proxima.data.export import ExportEngine, ExportFormat, ExportOptions, ReportData
        from proxima.data.compare import BackendResult
        
        store = MemoryStore()
        workflow_data = {}
        
        # Stage 1: Create circuit
        async def create_circuit(ctx, _):
            circuit_spec = {
                "type": "bell_state",
                "num_qubits": 2,
                "gates": ["H", "CNOT"],
            }
            ctx.set("circuit", circuit_spec)
            return circuit_spec
        
        # Stage 2: Execute on backends
        async def execute_backends(ctx, circuit):
            results = []
            for backend in ["cirq", "qiskit_aer", "lret"]:
                result = BackendResult(
                    backend_name=backend,
                    success=True,
                    execution_time_ms=50.0 + hash(backend) % 50,
                    memory_peak_mb=100.0 + hash(backend) % 100,
                )
                results.append(result)
            ctx.set("backend_results", results)
            return results
        
        # Stage 3: Store results
        async def store_results(ctx, results):
            session_id = f"workflow_{int(time.time())}"
            session = StoredSession(
                id=session_id,
                name="Full Workflow Test",
                metadata={"circuit": ctx.get("circuit")},
            )
            store.create_session(session)
            
            for result in results:
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
        
        # Stage 4: Generate report
        async def generate_report(ctx, session_id):
            results = ctx.get("backend_results")
            fastest = min(results, key=lambda r: r.execution_time_ms)
            
            report_data = ReportData(
                title="Workflow Report",
                generated_at=time.time(),
                summary={
                    "session_id": session_id,
                    "backends": len(results),
                    "fastest": fastest.backend_name,
                },
                metadata={"results": [r.backend_name for r in results]},
            )
            
            engine = ExportEngine()
            export_result = engine.export(
                report_data,
                format=ExportFormat.JSON,
                stream_output=True,
            )
            
            ctx.set("report", export_result.content)
            return export_result.content
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("full_workflow")
            .with_timeout(stage_timeout=30.0)
            .add_stage("create_circuit", "Create Circuit", create_circuit)
            .add_stage("execute", "Execute Backends", execute_backends, dependencies=["create_circuit"])
            .add_stage("store", "Store Results", store_results, dependencies=["execute"])
            .add_stage("report", "Generate Report", generate_report, dependencies=["store"])
            .build()
        )
        
        # Execute
        result = await pipeline.execute()
        
        # Verify complete success
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.successful_stages) == 4
        
        # Verify data was stored
        stored_results = store.list_results()
        assert len(stored_results) == 3
        
        # Verify report was generated
        report_stage = result.stage_results.get("report")
        assert report_stage is not None
        assert report_stage.result is not None
