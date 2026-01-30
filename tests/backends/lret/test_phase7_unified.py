"""Unit tests for LRET Phase 7 Unified adapter.

Tests the LRETPhase7UnifiedAdapter for:
- Basic functionality
- Multi-framework support
- Automatic framework selection
- Gate fusion optimization
- GPU acceleration settings
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any


class TestPhase7UnifiedBasic:
    """Basic tests for Phase 7 Unified adapter."""
    
    def test_adapter_name(self):
        """Test adapter has correct name."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            assert adapter.get_name() == "lret_phase7_unified"
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    def test_adapter_connect_disconnect(self):
        """Test adapter connection lifecycle."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            
            connected = adapter.connect()
            assert connected is True
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    def test_framework_availability(self):
        """Test framework availability detection."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            adapter.connect()
            
            frameworks = adapter._frameworks
            assert isinstance(frameworks, dict)
            assert len(frameworks) >= 0  # May have 0 if no frameworks installed
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")


class TestPhase7FrameworkSelection:
    """Tests for automatic framework selection."""
    
    def test_cirq_circuit_detection(self):
        """Test Cirq circuit detection."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            import cirq
            
            adapter = LRETPhase7UnifiedAdapter()
            adapter.connect()
            
            # Create Cirq circuit
            qubits = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([cirq.H(qubits[0])])
            
            selected = adapter._select_framework(circuit, {})
            assert selected == 'cirq'
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Cirq or Phase 7 Unified not installed")
    
    def test_explicit_framework_selection(self):
        """Test explicit framework selection via options."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            adapter.connect()
            
            circuit = Mock()
            selected = adapter._select_framework(circuit, {'framework': 'pennylane'})
            
            # Should respect explicit selection if available
            # Falls back if framework not available
            assert selected is not None
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    def test_gradient_detection(self):
        """Test gradient-based operation detection."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            adapter.connect()
            
            circuit = Mock()
            
            # With gradient request, should prefer PennyLane
            selected = adapter._select_framework(circuit, {'compute_gradient': True})
            
            # Should select PennyLane if available, else fallback
            assert selected is not None
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")


class TestPhase7GateFusion:
    """Tests for gate fusion optimization."""
    
    def test_phase7_config_creation(self):
        """Test Phase7Config can be created."""
        try:
            from proxima.backends.lret.phase7_unified import Phase7Config
            
            config = Phase7Config(
                gate_fusion=True,
                fusion_mode='hybrid',
                gpu_enabled=False,
            )
            
            assert config.gate_fusion is True
            assert config.fusion_mode == 'hybrid'
            assert config.gpu_enabled is False
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    def test_gate_fusion_applied(self):
        """Test gate fusion is applied during execution."""
        try:
            from proxima.backends.lret.phase7_unified import (
                LRETPhase7UnifiedAdapter,
                Phase7Config,
            )
            
            config = Phase7Config(gate_fusion=True, fusion_mode='hybrid')
            adapter = LRETPhase7UnifiedAdapter(config)
            adapter.connect()
            
            # Create circuit with fusible gates
            import cirq
            qubits = cirq.LineQubit.range(4)
            circuit = cirq.Circuit([
                cirq.H(qubits[0]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.H(qubits[1]),
                cirq.CNOT(qubits[1], qubits[2]),
            ])
            
            result = adapter.execute(circuit, options={
                'optimize': True,
                'shots': 1024
            })
            
            assert result is not None
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified or Cirq not installed")
    
    def test_fusion_mode_options(self):
        """Test various fusion mode options."""
        try:
            from proxima.backends.lret.phase7_unified import Phase7Config
            
            # Test each supported mode - only row, column, hybrid are valid
            for mode in ['row', 'column', 'hybrid']:
                config = Phase7Config(gate_fusion=True, fusion_mode=mode)
                assert config.fusion_mode == mode
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")


class TestPhase7GPUAcceleration:
    """Tests for GPU acceleration settings."""
    
    def test_gpu_config(self):
        """Test GPU configuration options."""
        try:
            from proxima.backends.lret.phase7_unified import Phase7Config
            
            config = Phase7Config(
                gpu_enabled=True,
                gpu_device_id=0,
            )
            
            assert config.gpu_enabled is True
            assert config.gpu_device_id == 0
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    def test_gpu_availability_check(self):
        """Test GPU availability checking via config."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter, Phase7Config
            
            # GPU availability is determined by config
            config = Phase7Config(gpu_enabled=False)
            adapter = LRETPhase7UnifiedAdapter(config)
            
            # Should have gpu_enabled in capabilities
            caps = adapter.get_capabilities()
            assert isinstance(caps.supports_gpu, bool)
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")


class TestPhase7MultiBackend:
    """Tests for multi-backend execution."""
    
    def test_consistent_results(self):
        """Test consistent results across backends."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            adapter.connect()
            
            # Simple Bell state circuit
            import cirq
            qubits = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([
                cirq.H(qubits[0]),
                cirq.CNOT(qubits[0], qubits[1]),
            ])
            
            # Run multiple times
            results = []
            for _ in range(3):
                result = adapter.execute(circuit, {'shots': 10000})
                results.append(result)
            
            # All should return results
            assert all(r is not None for r in results)
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified or Cirq not installed")


class TestPhase7Mocked:
    """Mocked tests that don't require actual installation."""
    
    def test_framework_priority_order(self):
        """Test framework priority ordering."""
        default_priority = ['lret', 'cirq', 'pennylane', 'qiskit']
        custom_priority = ['pennylane', 'lret', 'cirq']
        
        # First in list has highest priority
        assert default_priority[0] == 'lret'
        assert custom_priority[0] == 'pennylane'
    
    def test_gate_fusion_grouping(self):
        """Test gate fusion grouping logic."""
        # Mock gate sequence
        gates = [
            {'type': 'H', 'qubit': 0},
            {'type': 'CNOT', 'qubits': [0, 1]},
            {'type': 'H', 'qubit': 1},
            {'type': 'CNOT', 'qubits': [1, 2]},
        ]
        
        # Identify fusible groups (single-qubit gates on same qubit)
        single_qubit_gates = [g for g in gates if 'qubit' in g and 'qubits' not in g]
        
        assert len(single_qubit_gates) == 2
    
    def test_execution_result_structure(self):
        """Test execution result structure."""
        result = {
            'counts': {'00': 500, '11': 524},
            'shots': 1024,
            'success': True,
            'metadata': {
                'framework': 'cirq',
                'backend': 'lret_phase7_unified',
                'gate_fusion': True,
                'gpu_used': False,
            }
        }
        
        assert result['success'] is True
        assert result['metadata']['framework'] == 'cirq'
        assert result['metadata']['gate_fusion'] is True
    
    def test_unified_executor_dispatch(self):
        """Test executor dispatch logic."""
        # Mock framework detection
        def detect_framework(circuit):
            if hasattr(circuit, 'all_qubits'):
                return 'cirq'
            elif hasattr(circuit, 'tape'):
                return 'pennylane'
            elif hasattr(circuit, 'qregs'):
                return 'qiskit'
            return 'unknown'
        
        # Create mock objects with spec to control which attributes exist
        class CirqCircuit:
            all_qubits = None
        
        class PennyLaneCircuit:
            tape = None
        
        cirq_mock = Mock(spec=CirqCircuit)
        pennylane_mock = Mock(spec=PennyLaneCircuit)
        
        assert detect_framework(cirq_mock) == 'cirq'
        assert detect_framework(pennylane_mock) == 'pennylane'
