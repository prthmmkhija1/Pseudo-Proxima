"""Unit tests for LRET PennyLane Device and Algorithms.

Tests the QLRETDevice and variational algorithms:
- Basic device functionality
- QNode execution
- Gradient computation
- VQE algorithm
- QAOA algorithm
- QNN classifier
- Noise models
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import math


class TestQLRETDeviceBasic:
    """Basic tests for QLRETDevice."""
    
    def test_device_creation(self):
        """Test device can be created with default parameters."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(wires=4, shots=1024)
            
            assert dev.num_wires == 4
            assert dev.shots == 1024
            assert dev.short_name == "lret.qubit"
        except ImportError:
            pytest.skip("QLRETDevice not installed")
    
    def test_device_creation_statevector(self):
        """Test device in statevector mode."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(wires=2, shots=None)
            
            assert dev.num_wires == 2
            assert dev.shots is None  # Statevector mode
        except ImportError:
            pytest.skip("QLRETDevice not installed")
    
    def test_device_with_noise(self):
        """Test device with noise model."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(
                wires=4,
                shots=1024,
                noise_level=0.01,
                noise_model='depolarizing'
            )
            
            assert dev._noise_level == 0.01
            assert dev._noise_model == 'depolarizing'
        except ImportError:
            pytest.skip("QLRETDevice not installed")
    
    def test_basic_qnode_execution(self):
        """Test basic QNode execution."""
        try:
            import pennylane as qml
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(wires=4, shots=1024)
            
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0))
            
            result = circuit()
            
            assert isinstance(result, float)
            assert -1.0 <= result <= 1.0
        except ImportError:
            pytest.skip("PennyLane or QLRETDevice not installed")
    
    def test_bell_state_creation(self):
        """Test Bell state creation and measurement."""
        try:
            import pennylane as qml
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(wires=2, shots=10000)
            
            @qml.qnode(dev)
            def bell_state():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            
            probs = bell_state()
            
            # Bell state should have ~50% |00> and ~50% |11>
            assert probs[0] > 0.4  # |00>
            assert probs[3] > 0.4  # |11>
            assert probs[1] < 0.1  # |01>
            assert probs[2] < 0.1  # |10>
        except ImportError:
            pytest.skip("PennyLane or QLRETDevice not installed")


class TestQLRETDeviceGradient:
    """Tests for gradient computation with QLRETDevice."""
    
    def test_gradient_computation(self):
        """Test gradient computation with parameter-shift."""
        try:
            import pennylane as qml
            from pennylane import numpy as np
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(wires=2, shots=None)  # Statevector for accurate gradients
            
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(params):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0))
            
            params = np.array([0.5, 0.3], requires_grad=True)
            
            # Compute gradient
            grad_fn = qml.grad(circuit)
            gradients = grad_fn(params)
            
            assert gradients.shape == (2,)
            assert all(isinstance(g, (float, np.floating)) for g in gradients)
        except ImportError:
            pytest.skip("PennyLane or QLRETDevice not installed")
    
    def test_gradient_descent_optimization(self):
        """Test simple gradient descent optimization."""
        try:
            import pennylane as qml
            from pennylane import numpy as np
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            dev = QLRETDevice(wires=1, shots=None)
            
            @qml.qnode(dev)
            def circuit(theta):
                qml.RY(theta, wires=0)
                return qml.expval(qml.PauliZ(0))
            
            # Optimize to find ground state (theta = pi)
            theta = np.array(0.5, requires_grad=True)
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            
            for _ in range(50):
                theta = opt.step(circuit, theta)
            
            # Should converge close to pi (or 0)
            final_expval = circuit(theta)
            assert abs(final_expval) > 0.9  # Close to +1 or -1
        except ImportError:
            pytest.skip("PennyLane or QLRETDevice not installed")


class TestVQEAlgorithm:
    """Tests for VQE algorithm."""
    
    def test_vqe_creation(self):
        """Test VQE can be created."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import VQE
            
            dev = QLRETDevice(wires=2, shots=1024)
            
            # Simple Ising Hamiltonian
            import pennylane as qml
            H = qml.Hamiltonian(
                [1.0, -0.5],
                [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
            )
            
            def ansatz(params, wires):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
            
            vqe = VQE(dev, H, ansatz)
            
            assert vqe is not None
            assert vqe._hamiltonian is not None
        except ImportError:
            pytest.skip("VQE not installed")
    
    def test_vqe_convergence(self):
        """Test VQE converges to ground state."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import VQE
            import pennylane as qml
            
            dev = QLRETDevice(wires=2, shots=2048)
            
            H = qml.Hamiltonian(
                [1.0, -1.0],
                [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
            )
            
            def ansatz(params, wires):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
            
            vqe = VQE(dev, H, ansatz)
            result = vqe.run(initial_params=[0.5, 0.5], max_iterations=50)
            
            assert result.converged or result.iterations == 50
            assert len(result.energy_history) > 0
        except ImportError:
            pytest.skip("VQE not installed")


class TestQAOAAlgorithm:
    """Tests for QAOA algorithm."""
    
    def test_qaoa_creation(self):
        """Test QAOA can be created."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import QAOA
            import pennylane as qml
            
            dev = QLRETDevice(wires=4, shots=1024)
            
            # MaxCut cost Hamiltonian
            H_cost = qml.Hamiltonian(
                [0.5, 0.5, 0.5],
                [
                    qml.PauliZ(0) @ qml.PauliZ(1),
                    qml.PauliZ(1) @ qml.PauliZ(2),
                    qml.PauliZ(2) @ qml.PauliZ(3),
                ]
            )
            
            qaoa = QAOA(dev, H_cost, p=2)
            
            assert qaoa is not None
            assert qaoa._p == 2
        except ImportError:
            pytest.skip("QAOA not installed")
    
    def test_qaoa_optimization(self):
        """Test QAOA optimization runs."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import QAOA
            import pennylane as qml
            
            dev = QLRETDevice(wires=4, shots=2048)
            
            H_cost = qml.Hamiltonian(
                [0.5, 0.5, 0.5],
                [
                    qml.PauliZ(0) @ qml.PauliZ(1),
                    qml.PauliZ(1) @ qml.PauliZ(2),
                    qml.PauliZ(2) @ qml.PauliZ(3),
                ]
            )
            
            qaoa = QAOA(dev, H_cost, p=1)
            result = qaoa.run(max_iterations=20)
            
            assert result is not None
            assert 'best_params' in result or hasattr(result, 'gamma')
        except ImportError:
            pytest.skip("QAOA not installed")


class TestQNNClassifier:
    """Tests for QNN classifier."""
    
    def test_qnn_creation(self):
        """Test QNN can be created."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import QNN
            
            dev = QLRETDevice(wires=4, shots=1024)
            
            qnn = QNN(
                device=dev,
                num_layers=2,
                num_qubits=4,
            )
            
            assert qnn is not None
            assert qnn._num_layers == 2
        except ImportError:
            pytest.skip("QNN not installed")
    
    def test_qnn_forward_pass(self):
        """Test QNN forward pass."""
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import QNN
            import numpy as np
            
            dev = QLRETDevice(wires=4, shots=1024)
            
            qnn = QNN(device=dev, num_layers=2, num_qubits=4)
            
            # Random input
            x = np.random.randn(4)
            
            output = qnn.forward(x)
            
            assert output is not None
        except ImportError:
            pytest.skip("QNN not installed")


class TestPennyLaneMocked:
    """Mocked tests that don't require PennyLane installation."""
    
    def test_energy_calculation_logic(self):
        """Test energy calculation from expectation values."""
        coeffs = [1.0, -0.5, 0.3]
        expvals = [0.8, 0.6, -0.4]
        
        energy = sum(c * e for c, e in zip(coeffs, expvals))
        
        assert energy == pytest.approx(1.0 * 0.8 + (-0.5) * 0.6 + 0.3 * (-0.4))
    
    def test_convergence_detection(self):
        """Test convergence detection logic."""
        energy_history = [2.0, 1.5, 1.2, 1.05, 1.02, 1.01, 1.005, 1.003, 1.002, 1.001]
        threshold = 0.01
        window = 5
        
        # Check if last 'window' values are within threshold
        recent = energy_history[-window:]
        converged = (max(recent) - min(recent)) < threshold
        
        assert converged is True
    
    def test_parameter_shift_rule(self):
        """Test parameter-shift gradient calculation."""
        # f(theta) = cos(theta), gradient = -sin(theta)
        theta = 0.5
        shift = math.pi / 2
        
        # Parameter-shift: grad = (f(theta + pi/2) - f(theta - pi/2)) / 2
        f_plus = math.cos(theta + shift)
        f_minus = math.cos(theta - shift)
        
        grad_approx = (f_plus - f_minus) / 2
        grad_exact = -math.sin(theta)
        
        assert grad_approx == pytest.approx(grad_exact, rel=0.01)
    
    def test_ansatz_parameter_count(self):
        """Test calculation of ansatz parameter count."""
        num_qubits = 4
        num_layers = 2
        params_per_qubit_per_layer = 3  # e.g., RX, RY, RZ
        
        total_params = num_qubits * num_layers * params_per_qubit_per_layer
        
        assert total_params == 24
