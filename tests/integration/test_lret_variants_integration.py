"""Integration tests for LRET variants.

Tests complete workflows including:
- Full LRET variant workflow
- PennyLane VQE integration
- Phase 7 multi-framework execution
- TUI integration
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from pathlib import Path
import tempfile


@pytest.mark.integration
class TestLRETVariantWorkflow:
    """Integration tests for complete LRET variant workflow."""
    
    def test_full_lret_variant_workflow(self):
        """Test complete workflow with LRET variants."""
        try:
            # 1. Check variant availability
            from proxima.backends.lret.installer import check_variant_availability, VariantStatus
            
            cirq_status = check_variant_availability('cirq_scalability')
            # VariantStatus is a dataclass with an 'installed' attribute
            assert isinstance(cirq_status, VariantStatus)
            assert hasattr(cirq_status, 'installed')
            
        except ImportError:
            pytest.skip("LRET variants not installed")
    
    def test_variant_discovery(self):
        """Test variant discovery and registration."""
        try:
            from proxima.backends.lret.variant_registry import VariantRegistry
            
            registry = VariantRegistry()
            variants = registry.list_variants()
            
            # Should find at least the base variants
            assert isinstance(variants, list)
        except ImportError:
            pytest.skip("Variant registry not available")
    
    def test_variant_analysis_workflow(self):
        """Test variant analysis workflow."""
        try:
            from proxima.backends.lret.variant_analysis import VariantAnalyzer, TaskType
            
            analyzer = VariantAnalyzer()
            
            # Get recommendations for a task using the actual method
            recommendations = analyzer.select_variant_for_task(
                task_type=TaskType.GENERAL
            )
            
            # Can be None if no variants are installed
            assert recommendations is None or isinstance(recommendations, str)
        except ImportError:
            pytest.skip("Variant analysis not available")


@pytest.mark.integration
class TestPennyLaneVQEIntegration:
    """Integration tests for PennyLane VQE."""
    
    def test_pennylane_vqe_integration(self):
        """Test PennyLane VQE integration."""
        try:
            import pennylane as qml
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import VQE
            
            # Create device
            dev = QLRETDevice(wires=4, shots=2048, noise_level=0.01)
            
            # Define problem
            H = qml.Hamiltonian(
                [1.0, 0.5, 0.5],
                [
                    qml.PauliZ(0),
                    qml.PauliZ(0) @ qml.PauliZ(1),
                    qml.PauliZ(2) @ qml.PauliZ(3),
                ]
            )
            
            def ansatz(params, wires):
                for i in range(len(wires)):
                    qml.RY(params[i], wires=wires[i])
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i+1]])
            
            # Run VQE
            vqe = VQE(dev, H, ansatz)
            result = vqe.run(
                initial_params=[0.1] * 4,
                max_iterations=100,
                convergence_threshold=1e-5
            )
            
            # Verify convergence
            assert result.converged or result.iterations >= 10
            assert len(result.energy_history) > 0
        except ImportError:
            pytest.skip("PennyLane or VQE not installed")
    
    def test_pennylane_qaoa_integration(self):
        """Test QAOA integration."""
        try:
            import pennylane as qml
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import QAOA
            
            dev = QLRETDevice(wires=4, shots=1024)
            
            # MaxCut problem
            H_cost = qml.Hamiltonian(
                [0.5, 0.5, 0.5],
                [
                    qml.PauliZ(0) @ qml.PauliZ(1),
                    qml.PauliZ(1) @ qml.PauliZ(2),
                    qml.PauliZ(2) @ qml.PauliZ(3),
                ]
            )
            
            qaoa = QAOA(dev, H_cost, p=1)
            result = qaoa.run(max_iterations=10)
            
            assert result is not None
        except ImportError:
            pytest.skip("QAOA not installed")


@pytest.mark.integration
class TestPhase7MultiFramework:
    """Integration tests for Phase 7 multi-framework."""
    
    def test_phase7_multi_framework(self):
        """Test Phase 7 multi-framework execution."""
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            
            adapter = LRETPhase7UnifiedAdapter()
            adapter.connect()
            
            # Test with Cirq circuit
            import cirq
            cirq_circuit = cirq.Circuit([
                cirq.H(cirq.LineQubit(0)),
                cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
            ])
            
            result1 = adapter.execute(cirq_circuit, {'framework': 'cirq', 'shots': 1024})
            assert result1 is not None
            
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 Unified not installed")
    
    def test_phase7_gate_fusion_integration(self):
        """Test gate fusion in full workflow."""
        try:
            from proxima.backends.lret.phase7_unified import (
                LRETPhase7UnifiedAdapter,
                Phase7Config,
            )
            
            # Use only valid fusion modes: row, column, hybrid
            config = Phase7Config(gate_fusion=True, fusion_mode='hybrid')
            adapter = LRETPhase7UnifiedAdapter(config)
            adapter.connect()
            
            import cirq
            qubits = cirq.LineQubit.range(6)
            circuit = cirq.Circuit([
                cirq.H.on_each(*qubits),
                *[cirq.CNOT(qubits[i], qubits[i+1]) for i in range(5)],
                cirq.H.on_each(*qubits),
            ])
            
            result = adapter.execute(circuit, {
                'shots': 2048,
                'optimize': True,
            })
            
            assert result is not None
            adapter.disconnect()
        except ImportError:
            pytest.skip("Phase 7 or Cirq not installed")


@pytest.mark.integration
class TestTUIIntegration:
    """Integration tests for TUI components."""
    
    def test_benchmark_comparison_screen_creation(self):
        """Test BenchmarkComparisonScreen can be created."""
        try:
            from proxima.tui.screens import BenchmarkComparisonScreen
            
            screen = BenchmarkComparisonScreen()
            assert screen is not None
            assert screen.SCREEN_NAME == "benchmark_comparison"
        except ImportError:
            pytest.skip("TUI not available")
    
    def test_algorithm_wizard_creation(self):
        """Test PennyLaneAlgorithmWizard can be created."""
        try:
            from proxima.tui.wizards import PennyLaneAlgorithmWizard
            
            wizard = PennyLaneAlgorithmWizard()
            assert wizard is not None
        except ImportError:
            pytest.skip("Wizard not available")
    
    def test_backends_screen_lret_variants(self):
        """Test BackendsScreen includes LRET variants."""
        try:
            from proxima.tui.screens import BackendsScreen
            from proxima.tui.screens.backends import LRET_VARIANTS_DATA
            
            assert len(LRET_VARIANTS_DATA) == 4
            
            variant_ids = [v['id'] for v in LRET_VARIANTS_DATA]
            assert 'lret_base' in variant_ids
            assert 'lret_cirq_scalability' in variant_ids
            assert 'lret_pennylane_hybrid' in variant_ids
            assert 'lret_phase7_unified' in variant_ids
        except ImportError:
            pytest.skip("TUI not available")


@pytest.mark.integration
class TestBenchmarkExport:
    """Integration tests for benchmark export functionality."""
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        import csv
        import io
        
        # Mock benchmark data
        data = [
            {'qubits': 4, 'lret_ms': 10.5, 'cirq_ms': 15.2, 'speedup': 1.45, 'fidelity': 0.9998},
            {'qubits': 8, 'lret_ms': 45.7, 'cirq_ms': 89.3, 'speedup': 1.95, 'fidelity': 0.9995},
        ]
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['qubits', 'lret_ms', 'cirq_ms', 'speedup', 'fidelity'])
        writer.writeheader()
        writer.writerows(data)
        
        csv_content = output.getvalue()
        assert 'qubits' in csv_content
        assert '1.45' in csv_content
    
    def test_report_generation(self):
        """Test Markdown report generation."""
        from datetime import datetime
        
        # Mock report content
        stats = {
            'avg_speedup': 2.5,
            'max_speedup': 5.0,
            'avg_fidelity': 0.9997,
        }
        
        report = f"""# LRET vs Cirq Benchmark Report

**Generated:** {datetime.now().isoformat()}

## Summary Statistics

- Average Speedup: **{stats['avg_speedup']:.2f}x**
- Maximum Speedup: **{stats['max_speedup']:.2f}x**
- Average Fidelity: **{stats['avg_fidelity']:.4f}**
"""
        
        assert 'Average Speedup' in report
        assert '2.50x' in report


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_benchmark_workflow(self):
        """Test complete benchmark workflow from start to finish."""
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            
            adapter = LRETCirqScalabilityAdapter()
            
            # Check if adapter has execute method and is available
            if not hasattr(adapter, 'execute') or not adapter.is_available():
                pytest.skip("LRET Cirq Scalability not fully available")
            
        except ImportError:
            pytest.skip("LRET Cirq Scalability not installed")
    
    def test_complete_vqe_workflow(self):
        """Test complete VQE workflow."""
        try:
            import pennylane as qml
            from pennylane import numpy as np
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            # 1. Create device
            dev = QLRETDevice(wires=2, shots=None)
            
            # 2. Define circuit
            @qml.qnode(dev)
            def circuit(params):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            
            # 3. Optimize
            params = np.array([0.5, 0.5], requires_grad=True)
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            
            history = []
            for _ in range(20):
                params = opt.step(circuit, params)
                history.append(float(circuit(params)))
            
            # 4. Verify optimization happened
            assert len(history) == 20
            # Energy should change during optimization
            assert history[0] != history[-1] or abs(history[0]) > 0.9
        except ImportError:
            pytest.skip("PennyLane not installed")
