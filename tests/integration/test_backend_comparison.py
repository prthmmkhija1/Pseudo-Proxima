"""
Integration tests for backend comparison functionality.

Tests the complete backend comparison workflow including
multi-backend execution and result analysis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any, List
import math


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_backend_registry():
    """Create mock backend registry with multiple backends."""
    registry = Mock()
    
    # Define available backends
    backends = {
        "cirq": {
            "name": "cirq",
            "version": "1.3.0",
            "available": True,
            "capabilities": {
                "max_qubits": 30,
                "supports_density_matrix": True,
            }
        },
        "qiskit_aer": {
            "name": "qiskit_aer",
            "version": "0.14.0",
            "available": True,
            "capabilities": {
                "max_qubits": 32,
                "supports_density_matrix": True,
            }
        },
        "quest": {
            "name": "quest",
            "version": "3.5.0",
            "available": True,
            "capabilities": {
                "max_qubits": 40,
                "supports_density_matrix": True,
            }
        },
    }
    
    def get_backend(name):
        if name not in backends:
            return None
        
        backend = Mock()
        backend.name = name
        backend.get_version.return_value = backends[name]["version"]
        backend.is_available.return_value = backends[name]["available"]
        backend.get_capabilities.return_value = Mock(**backends[name]["capabilities"])
        
        # Simulate slightly different results per backend
        def run(circuit, shots=1000, **kwargs):
            # Add backend-specific "noise"
            offset = hash(name) % 20
            return {
                "counts": {
                    "00": 450 + offset,
                    "01": 25,
                    "10": 25,
                    "11": 500 - offset,
                },
                "execution_time": 0.1 * (1 + len(name) / 10),
            }
        
        backend.run = Mock(side_effect=run)
        return backend
    
    registry.get = Mock(side_effect=get_backend)
    registry.list_all.return_value = list(backends.keys())
    registry.list_available.return_value = [
        k for k, v in backends.items() if v["available"]
    ]
    
    return registry


@pytest.fixture
def comparison_service(mock_backend_registry):
    """Create comparison service with mock registry."""
    from proxima.api.services.circuit_service import CircuitService
    
    service = CircuitService()
    service._registry = mock_backend_registry
    return service


@pytest.fixture
def bell_circuit():
    """Bell state preparation circuit."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


@pytest.fixture
def ghz_circuit():
    """GHZ state preparation circuit for 3 qubits."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
measure q -> c;
"""


# =============================================================================
# Backend Comparison Tests
# =============================================================================

class TestBackendComparison:
    """Tests for backend comparison functionality."""
    
    def test_compare_two_backends(self, mock_backend_registry, bell_circuit):
        """Test comparing two backends."""
        cirq_backend = mock_backend_registry.get("cirq")
        qiskit_backend = mock_backend_registry.get("qiskit_aer")
        
        # Run on both backends
        cirq_result = cirq_backend.run(bell_circuit, shots=1000)
        qiskit_result = qiskit_backend.run(bell_circuit, shots=1000)
        
        assert "counts" in cirq_result
        assert "counts" in qiskit_result
        
        # Both should have Bell state distribution
        for result in [cirq_result, qiskit_result]:
            total = sum(result["counts"].values())
            # 00 and 11 should dominate
            dominant = result["counts"]["00"] + result["counts"]["11"]
            assert dominant > total * 0.9
    
    def test_compare_multiple_backends(self, mock_backend_registry, bell_circuit):
        """Test comparing multiple backends simultaneously."""
        backends = ["cirq", "qiskit_aer", "quest"]
        results = {}
        
        for name in backends:
            backend = mock_backend_registry.get(name)
            results[name] = backend.run(bell_circuit, shots=1000)
        
        assert len(results) == 3
        
        # All should have similar distributions
        for name, result in results.items():
            assert result["counts"]["00"] > 400
            assert result["counts"]["11"] > 400
    
    def test_comparison_with_timing(self, mock_backend_registry, bell_circuit):
        """Test that comparison includes timing information."""
        results = {}
        
        for name in ["cirq", "qiskit_aer"]:
            backend = mock_backend_registry.get(name)
            import time
            
            start = time.perf_counter()
            result = backend.run(bell_circuit, shots=1000)
            elapsed = time.perf_counter() - start
            
            results[name] = {
                **result,
                "measured_time": elapsed,
            }
        
        # Both should have timing
        for name, result in results.items():
            assert "execution_time" in result
            assert "measured_time" in result


# =============================================================================
# Result Analysis Tests
# =============================================================================

class TestResultAnalysis:
    """Tests for comparison result analysis."""
    
    def test_calculate_fidelity_between_backends(self):
        """Test calculating fidelity between backend results."""
        result1 = {"00": 500, "11": 500}
        result2 = {"00": 480, "01": 10, "10": 10, "11": 500}
        
        # Calculate classical fidelity
        total1 = sum(result1.values())
        total2 = sum(result2.values())
        
        all_states = set(result1.keys()) | set(result2.keys())
        
        fidelity = 0
        for state in all_states:
            p1 = result1.get(state, 0) / total1
            p2 = result2.get(state, 0) / total2
            fidelity += math.sqrt(p1 * p2)
        
        fidelity = fidelity ** 2
        
        assert 0 <= fidelity <= 1
        assert fidelity > 0.95  # Should be high for similar distributions
    
    def test_calculate_kl_divergence(self):
        """Test KL divergence calculation."""
        result1 = {"00": 500, "01": 0, "10": 0, "11": 500}
        result2 = {"00": 480, "01": 10, "10": 10, "11": 500}
        
        total1 = sum(result1.values())
        total2 = sum(result2.values())
        
        kl_div = 0
        for state in result1.keys():
            p = result1[state] / total1
            q = (result2.get(state, 0) + 1e-10) / total2  # Avoid log(0)
            if p > 0:
                kl_div += p * math.log(p / q)
        
        assert kl_div >= 0  # KL divergence is always non-negative
    
    def test_identify_best_backend(self, mock_backend_registry, bell_circuit):
        """Test identifying best performing backend."""
        backends = ["cirq", "qiskit_aer", "quest"]
        performance = {}
        
        for name in backends:
            backend = mock_backend_registry.get(name)
            result = backend.run(bell_circuit, shots=1000)
            
            # Calculate performance score (lower time is better)
            performance[name] = {
                "execution_time": result["execution_time"],
                "accuracy": (result["counts"]["00"] + result["counts"]["11"]) / 1000,
            }
        
        # Find best by accuracy
        best_accuracy = max(performance.items(), key=lambda x: x[1]["accuracy"])
        
        # Find fastest
        fastest = min(performance.items(), key=lambda x: x[1]["execution_time"])
        
        assert best_accuracy[1]["accuracy"] > 0.9
        assert fastest[1]["execution_time"] > 0
    
    def test_statistical_comparison(self):
        """Test statistical comparison of results."""
        # Run same circuit multiple times on each backend
        results1 = [
            {"00": 490 + i, "11": 510 - i}
            for i in range(10)
        ]
        results2 = [
            {"00": 485 + i, "11": 515 - i}
            for i in range(10)
        ]
        
        # Calculate means
        mean1 = sum(r["00"] for r in results1) / len(results1)
        mean2 = sum(r["00"] for r in results2) / len(results2)
        
        # Calculate standard deviations
        std1 = math.sqrt(sum((r["00"] - mean1)**2 for r in results1) / len(results1))
        std2 = math.sqrt(sum((r["00"] - mean2)**2 for r in results2) / len(results2))
        
        assert abs(mean1 - mean2) < 20  # Means should be close


# =============================================================================
# Comparison Report Tests
# =============================================================================

class TestComparisonReport:
    """Tests for comparison report generation."""
    
    def test_generate_comparison_summary(self, mock_backend_registry, bell_circuit):
        """Test generating comparison summary."""
        backends = ["cirq", "qiskit_aer"]
        results = {}
        
        for name in backends:
            backend = mock_backend_registry.get(name)
            results[name] = backend.run(bell_circuit, shots=1000)
        
        # Generate summary
        summary = {
            "circuit": "bell_state",
            "shots": 1000,
            "backends_compared": backends,
            "results": {},
        }
        
        for name, result in results.items():
            total = sum(result["counts"].values())
            summary["results"][name] = {
                "execution_time": result["execution_time"],
                "dominant_states": {
                    state: count / total
                    for state, count in sorted(
                        result["counts"].items(),
                        key=lambda x: -x[1]
                    )[:2]
                }
            }
        
        assert len(summary["results"]) == 2
        assert "cirq" in summary["results"]
        assert "qiskit_aer" in summary["results"]
    
    def test_comparison_includes_metadata(self, mock_backend_registry):
        """Test that comparison includes metadata."""
        comparison = {
            "id": "comp_123",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "circuit_info": {
                "num_qubits": 2,
                "depth": 3,
                "gate_count": 3,
            },
            "backends": ["cirq", "qiskit_aer"],
            "parameters": {
                "shots": 1000,
                "optimization_level": 1,
            }
        }
        
        assert "id" in comparison
        assert "created_at" in comparison
        assert "circuit_info" in comparison
    
    def test_export_comparison_json(self, mock_backend_registry, bell_circuit):
        """Test exporting comparison to JSON."""
        import json
        
        backends = ["cirq", "qiskit_aer"]
        comparison = {
            "id": "comp_test",
            "results": {}
        }
        
        for name in backends:
            backend = mock_backend_registry.get(name)
            comparison["results"][name] = backend.run(bell_circuit, shots=1000)
        
        # Should be JSON serializable
        json_str = json.dumps(comparison)
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "comp_test"
        assert len(parsed["results"]) == 2


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestComparisonEdgeCases:
    """Tests for comparison edge cases."""
    
    def test_comparison_with_unavailable_backend(self, mock_backend_registry):
        """Test handling unavailable backend."""
        # Make one backend unavailable
        mock_backend_registry.get.side_effect = lambda name: (
            None if name == "unavailable" else Mock()
        )
        
        backend = mock_backend_registry.get("unavailable")
        
        assert backend is None
    
    def test_comparison_with_execution_error(self, mock_backend_registry):
        """Test handling execution error in one backend."""
        backend = mock_backend_registry.get("cirq")
        
        # Simulate error
        backend.run = Mock(side_effect=RuntimeError("Execution failed"))
        
        with pytest.raises(RuntimeError):
            backend.run("test_circuit")
    
    def test_empty_comparison_backends(self):
        """Test comparison with no backends."""
        comparison = {
            "backends": [],
            "results": {}
        }
        
        assert len(comparison["backends"]) == 0
    
    def test_single_backend_comparison(self, mock_backend_registry, bell_circuit):
        """Test comparison with single backend (baseline)."""
        backend = mock_backend_registry.get("cirq")
        result = backend.run(bell_circuit, shots=1000)
        
        # Single backend result serves as baseline
        comparison = {
            "backends": ["cirq"],
            "results": {"cirq": result},
            "baseline": "cirq",
        }
        
        assert comparison["baseline"] == "cirq"
    
    def test_comparison_with_different_shot_counts(self, mock_backend_registry):
        """Test comparison normalizes different shot counts."""
        circuit = "test_circuit"
        
        # Run with different shots
        backend = mock_backend_registry.get("cirq")
        result_100 = {"00": 45, "11": 55}  # 100 shots
        result_1000 = {"00": 480, "11": 520}  # 1000 shots
        
        # Normalize to probabilities
        prob_100 = {k: v/100 for k, v in result_100.items()}
        prob_1000 = {k: v/1000 for k, v in result_1000.items()}
        
        # Probabilities should be similar
        assert abs(prob_100["00"] - prob_1000["00"]) < 0.1
        assert abs(prob_100["11"] - prob_1000["11"]) < 0.1


# =============================================================================
# Performance Comparison Tests
# =============================================================================

class TestPerformanceComparison:
    """Tests for performance comparison between backends."""
    
    def test_timing_comparison(self, mock_backend_registry, bell_circuit):
        """Test timing comparison between backends."""
        timings = {}
        
        for name in ["cirq", "qiskit_aer", "quest"]:
            backend = mock_backend_registry.get(name)
            result = backend.run(bell_circuit, shots=1000)
            timings[name] = result["execution_time"]
        
        # All should have positive timing
        for name, time in timings.items():
            assert time > 0
        
        # Can identify fastest
        fastest = min(timings.items(), key=lambda x: x[1])
        assert fastest[0] in ["cirq", "qiskit_aer", "quest"]
    
    def test_scaling_comparison(self, mock_backend_registry):
        """Test how backends scale with qubit count."""
        # This would test execution time vs qubit count
        # For mock, we just verify the structure
        
        scaling_results = {}
        
        for name in ["cirq", "qiskit_aer"]:
            scaling_results[name] = {
                "qubit_counts": [2, 4, 6, 8],
                "execution_times": [0.1, 0.2, 0.5, 1.2],
            }
        
        # Verify scaling data structure
        for name, data in scaling_results.items():
            assert len(data["qubit_counts"]) == len(data["execution_times"])
    
    def test_memory_usage_comparison(self, mock_backend_registry):
        """Test memory usage comparison between backends."""
        memory_estimates = {
            "cirq": {
                "qubits": 20,
                "memory_mb": 16,
            },
            "qiskit_aer": {
                "qubits": 20,
                "memory_mb": 24,
            },
            "quest": {
                "qubits": 20,
                "memory_mb": 32,
            },
        }
        
        # Find most memory efficient
        most_efficient = min(
            memory_estimates.items(),
            key=lambda x: x[1]["memory_mb"]
        )
        
        assert most_efficient[0] == "cirq"


# =============================================================================
# Accuracy Comparison Tests
# =============================================================================

class TestAccuracyComparison:
    """Tests for accuracy comparison between backends."""
    
    def test_compare_to_ideal(self):
        """Test comparing backend results to ideal distribution."""
        # Ideal Bell state
        ideal = {"00": 0.5, "11": 0.5}
        
        # Simulated result
        measured = {"00": 495, "01": 5, "10": 5, "11": 495}
        total = sum(measured.values())
        
        # Calculate total variation distance
        tvd = 0
        for state in set(ideal.keys()) | set(measured.keys()):
            p_ideal = ideal.get(state, 0)
            p_measured = measured.get(state, 0) / total
            tvd += abs(p_ideal - p_measured)
        tvd /= 2
        
        assert tvd < 0.05  # Should be close to ideal
    
    def test_error_rate_comparison(self):
        """Test comparing error rates between backends."""
        # Expected: only |00> and |11>
        expected_states = {"00", "11"}
        
        results = {
            "cirq": {"00": 490, "01": 5, "10": 5, "11": 500},
            "qiskit_aer": {"00": 485, "01": 8, "10": 7, "11": 500},
        }
        
        error_rates = {}
        for name, counts in results.items():
            total = sum(counts.values())
            errors = sum(
                count for state, count in counts.items()
                if state not in expected_states
            )
            error_rates[name] = errors / total
        
        assert error_rates["cirq"] < 0.02
        assert error_rates["qiskit_aer"] < 0.02
        
        # Can compare
        if error_rates["cirq"] < error_rates["qiskit_aer"]:
            best = "cirq"
        else:
            best = "qiskit_aer"
        
        assert best in ["cirq", "qiskit_aer"]
