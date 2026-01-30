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
    
    def test_adapter_name(self):
        """Test adapter has correct name."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            assert adapter.get_name() == "lret_cirq_scalability"
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    def test_adapter_capabilities(self):
        """Test adapter capabilities."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            
            caps = adapter.get_capabilities()
            assert caps.max_qubits == 20
            assert 'cirq_comparison' in caps.custom_features
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    def test_adapter_is_available(self):
        """Test adapter availability check."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            
            # is_available returns bool
            available = adapter.is_available()
            assert isinstance(available, bool)
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    def test_benchmark_result_structure(self):
        """Test benchmark result has correct structure."""
        try:
            from proxima.backends.lret.cirq_scalability import CirqScalabilityMetrics
            
            # Use the correct dataclass with correct field names
            result = CirqScalabilityMetrics(
                lret_time_ms=35.0,
                cirq_fdm_time_ms=55.0,
                speedup_factor=1.57,
                lret_final_rank=16,
                fidelity=0.9997,
                trace_distance=0.0003,
                qubit_count=8,
                circuit_depth=20,
            )
            
            assert result.qubit_count == 8
            assert result.circuit_depth == 20
            assert result.speedup_factor == pytest.approx(1.57, rel=0.01)
            assert result.fidelity > 0.999
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    def test_validate_circuit(self):
        """Test circuit validation."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            
            # Test with None circuit
            result = adapter.validate_circuit(None)
            assert result.valid is False
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
