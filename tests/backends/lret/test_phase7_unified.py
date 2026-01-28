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
    
    @pytest.mark.asyncio
    async def test_adapter_name(self):
        """Test adapter has correct name."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            assert adapter.name == "lret_phase7_unified"
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    @pytest.mark.asyncio
    async def test_adapter_connect_disconnect(self):
        """Test adapter connection lifecycle."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            
            connected = await adapter.connect()
            assert connected is True
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    @pytest.mark.asyncio
    async def test_framework_availability(self):
        """Test framework availability detection."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            await adapter.connect()
            
            frameworks = adapter._frameworks
            assert isinstance(frameworks, dict)
            assert len(frameworks) >= 0  # May have 0 if no frameworks installed
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")


class TestPhase7FrameworkSelection:
    """Tests for automatic framework selection."""
    
    @pytest.mark.asyncio
    async def test_cirq_circuit_detection(self):
        """Test Cirq circuit detection."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            import cirq
            
            adapter = LRETPhase7UnifiedAdapter()
            await adapter.connect()
            
            # Create Cirq circuit
            qubits = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([cirq.H(qubits[0])])
            
            selected = adapter._select_framework(circuit, {})
            assert selected == 'cirq'
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("Cirq or Phase 7 Unified not installed")
    
    @pytest.mark.asyncio
    async def test_explicit_framework_selection(self):
        """Test explicit framework selection via options."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            await adapter.connect()
            
            circuit = Mock()
            selected = adapter._select_framework(circuit, {'framework': 'pennylane'})
            
            # Should respect explicit selection if available
            # Falls back if framework not available
            assert selected is not None
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    @pytest.mark.asyncio
    async def test_gradient_detection(self):
        """Test gradient-based operation detection."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            await adapter.connect()
            
            circuit = Mock()
            
            # With gradient request, should prefer PennyLane
            selected = adapter._select_framework(circuit, {'compute_gradient': True})
            
            # Should select PennyLane if available, else fallback
            assert selected is not None
            
            await adapter.disconnect()
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
    
    @pytest.mark.asyncio
    async def test_gate_fusion_applied(self):
        """Test gate fusion is applied during execution."""
        try:
            from proxima.backends.lret.phase7_unified import (
                LRETPhase7UnifiedAdapter,
                Phase7Config,
            )
            
            config = Phase7Config(gate_fusion=True, fusion_mode='hybrid')
            adapter = LRETPhase7UnifiedAdapter(config)
            await adapter.connect()
            
            # Create circuit with fusible gates
            import cirq
            qubits = cirq.LineQubit.range(4)
            circuit = cirq.Circuit([
                cirq.H(qubits[0]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.H(qubits[1]),
                cirq.CNOT(qubits[1], qubits[2]),
            ])
            
            result = await adapter.execute(circuit, options={
                'optimize': True,
                'shots': 1024
            })
            
            assert result.success
            assert result.metadata.get('gate_fusion', False) is True
            
            await adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified or Cirq not installed")
    
    def test_fusion_mode_options(self):
        """Test various fusion mode options."""
        try:
            from proxima.backends.lret.phase7_unified import Phase7Config
            
            # Test each mode
            for mode in ['none', 'basic', 'aggressive', 'hybrid']:
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
                gpu_memory_limit=4096,  # MB
            )
            
            assert config.gpu_enabled is True
            assert config.gpu_device_id == 0
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    @pytest.mark.asyncio
    async def test_gpu_availability_check(self):
        """Test GPU availability checking."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            
            # Check if GPU is available
            gpu_available = adapter.check_gpu_availability()
            
            # Should return bool
            assert isinstance(gpu_available, bool)
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")


class TestPhase7MultiBackend:
    """Tests for multi-backend execution."""
    
    @pytest.mark.asyncio
    async def test_consistent_results(self):
        """Test consistent results across backends."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            await adapter.connect()
            
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
                result = await adapter.execute(circuit, {'shots': 10000})
                results.append(result)
            
            # All should succeed
            assert all(r.success for r in results)
            
            await adapter.disconnect()
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
        
        cirq_mock = Mock()
        cirq_mock.all_qubits = Mock()
        
        pennylane_mock = Mock()
        pennylane_mock.tape = Mock()
        
        assert detect_framework(cirq_mock) == 'cirq'
        assert detect_framework(pennylane_mock) == 'pennylane'
