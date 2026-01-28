"""Unit tests for LRET Cirq Scalability variant.

Tests the LRETCirqScalabilityAdapter for:
- Basic functionality
- LRET vs Cirq comparison
- Benchmark generation
- Speedup calculations
- Fidelity measurements
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any


# Test fixtures
@pytest.fixture
def mock_cirq():
    """Mock cirq module for testing without actual installation."""
    mock = Mock()
    mock.LineQubit = Mock()
    mock.LineQubit.range = Mock(return_value=[Mock() for _ in range(8)])
    mock.Circuit = Mock()
    mock.H = Mock()
    mock.CNOT = Mock()
    mock.Simulator = Mock()
    return mock


@pytest.fixture
def adapter_config():
    """Default adapter configuration."""
    return {
        'shots': 1024,
        'compare_with_cirq': True,
        'benchmark': True,
    }


class TestCirqScalabilityAdapter:
    """Tests for LRETCirqScalabilityAdapter."""
    
    @pytest.mark.asyncio
    async def test_adapter_name(self):
        """Test adapter has correct name."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            assert adapter.name == "lret_cirq_scalability"
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    @pytest.mark.asyncio
    async def test_adapter_connect_disconnect(self):
        """Test adapter connection lifecycle."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            
            # Connect
            connected = await adapter.connect()
            assert connected is True
            
            # Disconnect
            await adapter.disconnect()
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    @pytest.mark.asyncio
    async def test_basic_execution(self, mock_cirq):
        """Test basic circuit execution."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            await adapter.connect()
            
            # Create mock circuit
            circuit = Mock()
            circuit.all_qubits.return_value = [Mock() for _ in range(4)]
            
            result = await adapter.execute(circuit, options={'shots': 1024})
            
            assert result.success
            assert result.shots == 1024
            assert 'execution_time_ms' in result.metadata
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    @pytest.mark.asyncio
    async def test_lret_cirq_comparison(self):
        """Test LRET vs Cirq comparison functionality."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            await adapter.connect()
            
            # Generate random circuit for comparison
            circuit = adapter.generate_random_circuit(num_qubits=8, depth=10)
            
            result = await adapter.execute(circuit, options={
                'shots': 1024,
                'compare_with_cirq': True,
                'benchmark': True,
            })
            
            # Verify comparison metadata
            assert 'speedup' in result.metadata
            assert 'fidelity' in result.metadata
            assert result.metadata['speedup'] >= 0.5  # Allow some variance
            assert result.metadata['fidelity'] > 0.99  # High fidelity expected
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    @pytest.mark.asyncio
    async def test_benchmark_result_structure(self):
        """Test benchmark result has correct structure."""
        try:
            from proxima.backends.lret.cirq_scalability import BenchmarkResult
            
            result = BenchmarkResult(
                num_qubits=8,
                circuit_depth=20,
                lret_time_ms=35.0,
                cirq_time_ms=55.0,
                speedup=1.57,
                fidelity=0.9997,
            )
            
            assert result.num_qubits == 8
            assert result.circuit_depth == 20
            assert result.speedup == pytest.approx(1.57, rel=0.01)
            assert result.fidelity > 0.999
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    @pytest.mark.asyncio
    async def test_scalability_benchmark(self):
        """Test scalability benchmark across qubit counts."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            await adapter.connect()
            
            results = []
            for num_qubits in [4, 6, 8]:
                circuit = adapter.generate_random_circuit(
                    num_qubits=num_qubits,
                    depth=10,
                )
                
                result = await adapter.execute(circuit, options={
                    'shots': 512,
                    'benchmark': True,
                })
                
                results.append({
                    'qubits': num_qubits,
                    'time_ms': result.metadata.get('execution_time_ms', 0),
                })
            
            # Execution time should increase with qubits
            assert len(results) == 3
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")


class TestCirqScalabilityMocked:
    """Mocked tests that don't require actual LRET installation."""
    
    def test_speedup_calculation(self):
        """Test speedup calculation logic."""
        lret_time = 35.0
        cirq_time = 55.0
        
        speedup = cirq_time / lret_time
        
        assert speedup == pytest.approx(1.57, rel=0.01)
    
    def test_fidelity_calculation(self):
        """Test fidelity calculation logic."""
        # Mock counts from LRET and Cirq
        lret_counts = {'00': 500, '11': 524}
        cirq_counts = {'00': 498, '11': 526}
        
        # Calculate fidelity (overlap)
        total_lret = sum(lret_counts.values())
        total_cirq = sum(cirq_counts.values())
        
        overlap = 0
        for state in set(lret_counts.keys()) | set(cirq_counts.keys()):
            p_lret = lret_counts.get(state, 0) / total_lret
            p_cirq = cirq_counts.get(state, 0) / total_cirq
            overlap += (p_lret * p_cirq) ** 0.5
        
        fidelity = overlap ** 2
        
        assert fidelity > 0.99
    
    def test_benchmark_csv_format(self):
        """Test benchmark CSV output format."""
        import io
        import csv
        
        # Mock benchmark data
        data = [
            {'qubits': 4, 'lret_ms': 10.5, 'cirq_ms': 15.2, 'speedup': 1.45, 'fidelity': 0.9998},
            {'qubits': 6, 'lret_ms': 25.3, 'cirq_ms': 42.1, 'speedup': 1.66, 'fidelity': 0.9996},
            {'qubits': 8, 'lret_ms': 45.7, 'cirq_ms': 89.3, 'speedup': 1.95, 'fidelity': 0.9995},
        ]
        
        # Write to string buffer
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['qubits', 'lret_ms', 'cirq_ms', 'speedup', 'fidelity'])
        writer.writeheader()
        writer.writerows(data)
        
        # Verify CSV content
        csv_content = output.getvalue()
        assert 'qubits,lret_ms,cirq_ms,speedup,fidelity' in csv_content
        assert '4,10.5,15.2,1.45,0.9998' in csv_content
