"""Performance tests for Agent modules.

Phase 10: Integration & Testing

Tests cover:
- High-frequency operations (100 commands)
- Large output handling (100K lines)
- Concurrent operations (10 parallel tools)
- Long-running operations
- Memory usage benchmarks
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockLLMClient,
    MockLLMResponse,
    MockSubprocessFactory,
    MockProcessOutput,
    MockTelemetry,
    create_mock_llm_client,
    create_mock_subprocess_factory,
    create_mock_telemetry,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fast_llm_client():
    """Create fast LLM client for performance testing."""
    return create_mock_llm_client(latency_ms=10.0)


@pytest.fixture
def mock_subprocess_factory():
    """Create subprocess factory for performance testing."""
    return create_mock_subprocess_factory()


@pytest.fixture
def mock_telemetry():
    """Create telemetry for performance testing."""
    return create_mock_telemetry()


# =============================================================================
# HIGH-FREQUENCY OPERATION TESTS
# =============================================================================

class TestHighFrequencyOperations:
    """Tests for high-frequency operations performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_100_sequential_commands(self, mock_subprocess_factory):
        """Test executing 100 sequential commands within time limit."""
        start_time = time.time()
        
        for i in range(100):
            mock_subprocess_factory.set_output(
                f"cmd_{i}",
                MockProcessOutput(stdout=f"output_{i}\n", returncode=0, duration_ms=1)
            )
            
            process = await mock_subprocess_factory.create_subprocess_shell(f"cmd_{i}")
            await process.wait()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete within 5 seconds (generous for mocks)
        assert elapsed_ms < 5000, f"100 commands took {elapsed_ms}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_100_llm_requests(self, fast_llm_client):
        """Test 100 LLM requests performance."""
        start_time = time.time()
        
        for i in range(100):
            fast_llm_client.queue_response(MockLLMResponse(
                content=f"Response {i}"
            ))
            await fast_llm_client.chat([
                {"role": "user", "content": f"Message {i}"}
            ])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Average latency should be under 20ms per request
        avg_latency = elapsed_ms / 100
        assert avg_latency < 20, f"Average latency {avg_latency}ms exceeds 20ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_rapid_telemetry_recording(self, mock_telemetry):
        """Test rapid telemetry recording performance."""
        start_time = time.time()
        
        for i in range(1000):
            mock_telemetry.record_event(
                event_type="performance_test",
                data={"index": i, "timestamp": time.time()}
            )
            mock_telemetry.record_metric(f"metric_{i % 10}", float(i))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete within 100ms
        assert elapsed_ms < 100, f"1000 telemetry records took {elapsed_ms}ms"
        assert len(mock_telemetry.events) == 1000


# =============================================================================
# LARGE OUTPUT HANDLING TESTS
# =============================================================================

class TestLargeOutputHandling:
    """Tests for handling large output volumes."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_100k_lines_output(self, mock_subprocess_factory):
        """Test handling 100K lines of output."""
        # Generate large output
        large_output = "\n".join([f"Line {i}: " + "x" * 80 for i in range(100000)])
        
        mock_subprocess_factory.set_output(
            "large_output",
            MockProcessOutput(stdout=large_output, returncode=0, duration_ms=100)
        )
        
        start_time = time.time()
        
        process = await mock_subprocess_factory.create_subprocess_shell("large_output")
        stdout, _ = await process.communicate()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should handle large output within 1 second
        assert elapsed_ms < 1000, f"100K lines took {elapsed_ms}ms"
        
        # Verify output integrity
        output_str = stdout.decode()
        assert "Line 0:" in output_str
        assert "Line 99999:" in output_str
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_llm_response(self, fast_llm_client):
        """Test handling large LLM response."""
        # Generate large response
        large_content = "A" * 50000  # 50KB response
        
        fast_llm_client.queue_response(MockLLMResponse(
            content=large_content,
            completion_tokens=10000,
        ))
        
        start_time = time.time()
        
        response = await fast_llm_client.chat([
            {"role": "user", "content": "Generate large output"}
        ])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should handle within 100ms
        assert elapsed_ms < 100
        assert len(response["choices"][0]["message"]["content"]) == 50000


# =============================================================================
# CONCURRENT OPERATIONS TESTS
# =============================================================================

class TestConcurrentOperations:
    """Tests for concurrent operations performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_10_parallel_commands(self, mock_subprocess_factory):
        """Test 10 parallel command executions."""
        # Configure commands
        for i in range(10):
            mock_subprocess_factory.set_output(
                f"parallel_{i}",
                MockProcessOutput(stdout=f"Result {i}\n", returncode=0, duration_ms=50)
            )
        
        start_time = time.time()
        
        # Start all commands
        processes = [
            await mock_subprocess_factory.create_subprocess_shell(f"parallel_{i}")
            for i in range(10)
        ]
        
        # Wait for all
        results = await asyncio.gather(*[p.wait() for p in processes])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Parallel execution should be faster than sequential
        # 10 x 50ms = 500ms sequential, should be much less for parallel
        assert elapsed_ms < 500, f"10 parallel commands took {elapsed_ms}ms"
        assert all(r == 0 for r in results)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_llm_and_commands(self, fast_llm_client, mock_subprocess_factory):
        """Test concurrent LLM calls and commands."""
        mock_subprocess_factory.set_output(
            "concurrent_cmd",
            MockProcessOutput(stdout="cmd output\n", returncode=0, duration_ms=10)
        )
        
        start_time = time.time()
        
        async def llm_task():
            fast_llm_client.queue_response(MockLLMResponse(content="LLM response"))
            return await fast_llm_client.chat([{"role": "user", "content": "test"}])
        
        async def cmd_task():
            process = await mock_subprocess_factory.create_subprocess_shell("concurrent_cmd")
            await process.wait()
            return process.returncode
        
        # Run 5 of each concurrently
        tasks = []
        for _ in range(5):
            tasks.append(llm_task())
            tasks.append(cmd_task())
        
        results = await asyncio.gather(*tasks)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete within 500ms
        assert elapsed_ms < 500, f"Concurrent operations took {elapsed_ms}ms"
        assert len(results) == 10


# =============================================================================
# MEMORY USAGE TESTS
# =============================================================================

class TestMemoryUsage:
    """Tests for memory usage benchmarks."""
    
    @pytest.mark.performance
    def test_telemetry_memory_growth(self, mock_telemetry):
        """Test telemetry doesn't cause excessive memory growth."""
        import sys
        
        initial_events = len(mock_telemetry.events)
        
        # Record many events
        for i in range(10000):
            mock_telemetry.record_event(
                "memory_test",
                {"data": "x" * 100, "index": i}
            )
        
        # Check event count
        assert len(mock_telemetry.events) == initial_events + 10000
        
        # Clear to test cleanup
        mock_telemetry.clear()
        assert len(mock_telemetry.events) == 0
    
    @pytest.mark.performance
    def test_large_data_structure_handling(self):
        """Test handling large data structures."""
        # Create large data structure
        large_data = {
            f"key_{i}": {"nested": {"data": [j for j in range(100)]}}
            for i in range(1000)
        }
        
        import sys
        size = sys.getsizeof(str(large_data))
        
        # Should be manageable size (under 50MB)
        assert size < 50 * 1024 * 1024


# =============================================================================
# LATENCY BENCHMARKS
# =============================================================================

class TestLatencyBenchmarks:
    """Tests for operation latency benchmarks."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_command_overhead(self, mock_subprocess_factory):
        """Test command execution overhead is under 100ms."""
        mock_subprocess_factory.set_output(
            "quick",
            MockProcessOutput(stdout="", returncode=0, duration_ms=0)
        )
        
        latencies = []
        
        for _ in range(50):
            start = time.time()
            process = await mock_subprocess_factory.create_subprocess_shell("quick")
            await process.wait()
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Average overhead should be under 10ms, max under 100ms
        assert avg_latency < 10, f"Average latency {avg_latency}ms"
        assert max_latency < 100, f"Max latency {max_latency}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_llm_request_overhead(self, fast_llm_client):
        """Test LLM request overhead."""
        latencies = []
        
        for _ in range(50):
            fast_llm_client.queue_response(MockLLMResponse(content="Quick"))
            
            start = time.time()
            await fast_llm_client.chat([{"role": "user", "content": "test"}])
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Configured 10ms latency + overhead should be under 20ms average
        assert avg_latency < 20, f"Average LLM latency {avg_latency}ms"


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for agent components."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_load(self, fast_llm_client, mock_subprocess_factory):
        """Test sustained load over time."""
        mock_subprocess_factory.set_output(
            "stress",
            MockProcessOutput(returncode=0, duration_ms=1)
        )
        
        start_time = time.time()
        operations = 0
        
        # Run for 2 seconds
        while time.time() - start_time < 2.0:
            fast_llm_client.queue_response(MockLLMResponse(content="ok"))
            await fast_llm_client.chat([{"role": "user", "content": "stress"}])
            
            process = await mock_subprocess_factory.create_subprocess_shell("stress")
            await process.wait()
            
            operations += 2
        
        elapsed = time.time() - start_time
        ops_per_second = operations / elapsed
        
        # Should sustain at least 50 operations per second
        assert ops_per_second > 50, f"Only {ops_per_second} ops/sec"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_burst_load(self, mock_subprocess_factory):
        """Test handling burst of commands."""
        for i in range(100):
            mock_subprocess_factory.set_output(
                f"burst_{i}",
                MockProcessOutput(returncode=0, duration_ms=1)
            )
        
        start_time = time.time()
        
        # Create all processes at once
        processes = []
        for i in range(100):
            process = await mock_subprocess_factory.create_subprocess_shell(f"burst_{i}")
            processes.append(process)
        
        # Wait for all
        await asyncio.gather(*[p.wait() for p in processes])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Burst should complete quickly
        assert elapsed_ms < 1000, f"Burst took {elapsed_ms}ms"


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def benchmark(func, iterations: int = 100) -> dict:
    """Run benchmark and return statistics."""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        func()
        times.append((time.time() - start) * 1000)
    
    return {
        "min_ms": min(times),
        "max_ms": max(times),
        "avg_ms": sum(times) / len(times),
        "total_ms": sum(times),
        "iterations": iterations,
    }


class TestBenchmarkUtilities:
    """Tests for benchmark utility functions."""
    
    def test_benchmark_function(self):
        """Test benchmark utility."""
        def simple_func():
            _ = [i * 2 for i in range(100)]
        
        result = benchmark(simple_func, iterations=50)
        
        assert "min_ms" in result
        assert "max_ms" in result
        assert "avg_ms" in result
        assert result["iterations"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
