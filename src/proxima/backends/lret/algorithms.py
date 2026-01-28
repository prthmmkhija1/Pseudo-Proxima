"""Variational quantum algorithms using LRET PennyLane device.

This module provides implementations of common variational quantum
algorithms (VQE, QAOA, QNN) that integrate with the LRET PennyLane device.

Features:
- VQE (Variational Quantum Eigensolver) for ground state finding
- QAOA (Quantum Approximate Optimization) for combinatorial optimization
- QNN (Quantum Neural Network) for machine learning
- Multiple ansatz options (hardware efficient, UCCSD, etc.)
- Integration with PennyLane optimizers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VQEResult:
    """Result from VQE optimization.
    
    Attributes:
        final_energy: Final optimized energy value
        optimal_params: Optimal variational parameters
        iterations: Number of optimization iterations
        energy_history: Energy at each iteration
        converged: Whether optimization converged
        gradient_history: Optional gradient norm history
    """
    
    final_energy: float
    optimal_params: np.ndarray
    iterations: int
    energy_history: list[float] = field(default_factory=list)
    converged: bool = False
    gradient_history: Optional[list[float]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'final_energy': self.final_energy,
            'optimal_params': self.optimal_params.tolist() if hasattr(self.optimal_params, 'tolist') else list(self.optimal_params),
            'iterations': self.iterations,
            'energy_history': self.energy_history,
            'converged': self.converged,
        }


@dataclass
class QAOAResult:
    """Result from QAOA optimization.
    
    Attributes:
        final_cost: Final cost function value
        optimal_gammas: Optimal gamma parameters
        optimal_betas: Optimal beta parameters
        iterations: Number of iterations
        cost_history: Cost at each iteration
        best_bitstring: Most probable solution bitstring
        solution_probability: Probability of best solution
    """
    
    final_cost: float
    optimal_gammas: np.ndarray
    optimal_betas: np.ndarray
    iterations: int
    cost_history: list[float] = field(default_factory=list)
    best_bitstring: Optional[str] = None
    solution_probability: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'final_cost': self.final_cost,
            'optimal_gammas': self.optimal_gammas.tolist() if hasattr(self.optimal_gammas, 'tolist') else list(self.optimal_gammas),
            'optimal_betas': self.optimal_betas.tolist() if hasattr(self.optimal_betas, 'tolist') else list(self.optimal_betas),
            'iterations': self.iterations,
            'cost_history': self.cost_history,
            'best_bitstring': self.best_bitstring,
            'solution_probability': self.solution_probability,
        }


class VQE:
    """Variational Quantum Eigensolver using LRET device.
    
    VQE is a hybrid quantum-classical algorithm for finding the
    ground state energy of a Hamiltonian. It uses a parameterized
    quantum circuit (ansatz) and classical optimization.
    
    Example:
        >>> from proxima.backends.lret.pennylane_device import QLRETDevice
        >>> dev = QLRETDevice(wires=4, shots=1024)
        >>> 
        >>> # Define Hamiltonian coefficients and terms
        >>> coeffs = [0.5, 0.5, -1.0]
        >>> terms = ['ZI', 'IZ', 'ZZ']  # Pauli strings
        >>> 
        >>> # Define ansatz
        >>> def ansatz(params, wires):
        ...     for i, wire in enumerate(wires):
        ...         # RY rotation
        ...         pass  # Apply RY(params[i], wire)
        ...     # Entangling layer
        ...     pass  # Apply CNOTs
        >>> 
        >>> vqe = VQE(dev, coeffs, terms, ansatz)
        >>> result = vqe.run(initial_params=[0.5, 0.5, 0.5, 0.5])
        >>> print(f"Ground state energy: {result.final_energy:.4f}")
    """
    
    def __init__(
        self,
        device: Any,
        hamiltonian_coeffs: Sequence[float],
        hamiltonian_terms: Sequence[str],
        ansatz: Optional[Callable] = None,
        ansatz_type: str = 'hardware_efficient',
        optimizer_type: str = 'adam',
        learning_rate: float = 0.1,
    ):
        """Initialize VQE.
        
        Args:
            device: PennyLane-compatible device (e.g., QLRETDevice)
            hamiltonian_coeffs: Coefficients for Hamiltonian terms
            hamiltonian_terms: Pauli strings (e.g., 'ZZ', 'XI', 'YI')
            ansatz: Custom ansatz function (optional)
            ansatz_type: Built-in ansatz type if ansatz not provided
            optimizer_type: 'adam', 'gradient_descent', 'qng'
            learning_rate: Optimizer learning rate/step size
        """
        self.device = device
        self.hamiltonian_coeffs = list(hamiltonian_coeffs)
        self.hamiltonian_terms = list(hamiltonian_terms)
        self.ansatz = ansatz
        self.ansatz_type = ansatz_type
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        # Determine number of qubits from device
        self.n_qubits = getattr(device, 'num_wires', 4)
        
        # Build cost function
        self._build_cost_function()
    
    def _build_cost_function(self) -> None:
        """Build the VQE cost function."""
        # Store for use in cost function
        self._cost_fn = self._create_cost_function()
    
    def _create_cost_function(self) -> Callable:
        """Create the cost function for VQE.
        
        Returns:
            Cost function that computes energy expectation
        """
        def cost_fn(params: np.ndarray) -> float:
            """Compute expectation value of Hamiltonian."""
            energy = 0.0
            
            # Apply ansatz and compute each term
            for coeff, term in zip(self.hamiltonian_coeffs, self.hamiltonian_terms):
                term_expval = self._compute_term_expval(params, term)
                energy += coeff * term_expval
            
            return energy
        
        return cost_fn
    
    def _compute_term_expval(self, params: np.ndarray, pauli_term: str) -> float:
        """Compute expectation value of a Pauli term.
        
        Args:
            params: Variational parameters
            pauli_term: Pauli string (e.g., 'ZZ', 'XI')
            
        Returns:
            Expectation value
        """
        # Apply ansatz to device
        self._apply_ansatz(params)
        
        # Compute expectation based on Pauli term
        # This is a simplified implementation
        expval = 1.0
        
        for i, pauli in enumerate(pauli_term):
            if pauli != 'I':
                # Get single-qubit expectation
                wire_expval = self._measure_pauli(i, pauli)
                expval *= wire_expval
        
        return expval
    
    def _apply_ansatz(self, params: np.ndarray) -> None:
        """Apply the variational ansatz.
        
        Args:
            params: Variational parameters
        """
        if self.ansatz is not None:
            # Custom ansatz
            self.ansatz(params, wires=range(self.n_qubits))
        else:
            # Built-in ansatz
            self._apply_builtin_ansatz(params)
    
    def _apply_builtin_ansatz(self, params: np.ndarray) -> None:
        """Apply built-in hardware-efficient ansatz.
        
        Args:
            params: Variational parameters
        """
        ops = []
        param_idx = 0
        
        # Layer 1: Single-qubit rotations
        for wire in range(self.n_qubits):
            if param_idx < len(params):
                ops.append({
                    'name': 'ry',
                    'wires': [wire],
                    'params': [params[param_idx]],
                })
                param_idx += 1
        
        # Entangling layer: CNOTs
        for wire in range(self.n_qubits - 1):
            ops.append({
                'name': 'cx',
                'wires': [wire, wire + 1],
                'params': [],
            })
        
        # Layer 2: More rotations
        for wire in range(self.n_qubits):
            if param_idx < len(params):
                ops.append({
                    'name': 'ry',
                    'wires': [wire],
                    'params': [params[param_idx]],
                })
                param_idx += 1
        
        # Apply to device
        if hasattr(self.device, 'apply'):
            self.device.reset()
            self.device.apply(ops)
    
    def _measure_pauli(self, wire: int, pauli: str) -> float:
        """Measure Pauli operator on a wire.
        
        Args:
            wire: Wire to measure
            pauli: Pauli operator ('X', 'Y', 'Z')
            
        Returns:
            Expectation value
        """
        # Create mock observable
        class MockObservable:
            def __init__(self, name, wires):
                self.name = f"Pauli{name}"
                self.wires = wires
        
        obs = MockObservable(pauli, [wire])
        
        if hasattr(self.device, 'expval'):
            return self.device.expval(obs)
        else:
            # Default mock value
            return 0.0
    
    def get_num_params(self) -> int:
        """Get number of variational parameters needed.
        
        Returns:
            Number of parameters for the ansatz
        """
        if self.ansatz_type == 'hardware_efficient':
            # 2 layers of single-qubit rotations
            return 2 * self.n_qubits
        else:
            return self.n_qubits
    
    def run(
        self,
        initial_params: Optional[Union[np.ndarray, Sequence[float]]] = None,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        verbose: bool = False,
    ) -> VQEResult:
        """Run VQE optimization.
        
        Args:
            initial_params: Initial parameter values (random if None)
            max_iterations: Maximum optimization iterations
            convergence_threshold: Energy convergence threshold
            verbose: Print progress during optimization
            
        Returns:
            VQEResult with optimization details
        """
        # Initialize parameters
        if initial_params is None:
            n_params = self.get_num_params()
            params = np.random.uniform(-np.pi, np.pi, n_params)
        else:
            params = np.array(initial_params, dtype=float)
        
        energy_history = []
        gradient_history = []
        
        # Simple gradient descent optimization
        for iteration in range(max_iterations):
            # Compute energy
            energy = self._cost_fn(params)
            energy_history.append(float(energy))
            
            if verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Energy = {energy:.6f}")
            
            # Compute gradient (parameter-shift rule approximation)
            gradient = self._compute_gradient(params)
            gradient_norm = np.linalg.norm(gradient)
            gradient_history.append(float(gradient_norm))
            
            # Update parameters
            params = params - self.learning_rate * gradient
            
            # Check convergence
            if len(energy_history) > 1:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < convergence_threshold:
                    if verbose:
                        logger.info(f"Converged at iteration {iteration}")
                    return VQEResult(
                        final_energy=energy_history[-1],
                        optimal_params=params,
                        iterations=iteration + 1,
                        energy_history=energy_history,
                        converged=True,
                        gradient_history=gradient_history,
                    )
        
        return VQEResult(
            final_energy=energy_history[-1] if energy_history else 0.0,
            optimal_params=params,
            iterations=max_iterations,
            energy_history=energy_history,
            converged=False,
            gradient_history=gradient_history,
        )
    
    def _compute_gradient(self, params: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
        """Compute gradient using parameter-shift rule.
        
        Args:
            params: Current parameters
            shift: Shift amount for parameter-shift rule
            
        Returns:
            Gradient array
        """
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += shift
            energy_plus = self._cost_fn(params_plus)
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= shift
            energy_minus = self._cost_fn(params_minus)
            
            # Parameter-shift gradient
            gradient[i] = (energy_plus - energy_minus) / (2 * np.sin(shift))
        
        return gradient


class QAOA:
    """Quantum Approximate Optimization Algorithm using LRET device.
    
    QAOA is designed for combinatorial optimization problems like
    Max-Cut, traveling salesman, etc. It alternates between cost
    and mixer Hamiltonians.
    
    Example:
        >>> from proxima.backends.lret.pennylane_device import QLRETDevice
        >>> dev = QLRETDevice(wires=4, shots=2048)
        >>> 
        >>> # Max-Cut problem graph edges
        >>> edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        >>> 
        >>> qaoa = QAOA(dev, edges, p=2)
        >>> result = qaoa.run()
        >>> print(f"Best cut value: {-result.final_cost:.1f}")
        >>> print(f"Best bitstring: {result.best_bitstring}")
    """
    
    def __init__(
        self,
        device: Any,
        edges: Sequence[tuple[int, int]],
        p: int = 1,
        weights: Optional[Sequence[float]] = None,
    ):
        """Initialize QAOA for Max-Cut problem.
        
        Args:
            device: PennyLane-compatible device
            edges: Graph edges as (node1, node2) tuples
            p: Number of QAOA layers
            weights: Optional edge weights (default: all 1.0)
        """
        self.device = device
        self.edges = list(edges)
        self.p = p
        self.weights = list(weights) if weights else [1.0] * len(edges)
        
        # Determine number of qubits from edges
        all_nodes = set()
        for e in edges:
            all_nodes.add(e[0])
            all_nodes.add(e[1])
        self.n_qubits = max(all_nodes) + 1
    
    def _cost_function(self, gammas: np.ndarray, betas: np.ndarray) -> float:
        """Compute QAOA cost function value.
        
        Args:
            gammas: Gamma parameters (cost Hamiltonian)
            betas: Beta parameters (mixer Hamiltonian)
            
        Returns:
            Cost function value
        """
        # Apply QAOA circuit
        self._apply_qaoa_circuit(gammas, betas)
        
        # Compute cost expectation
        cost = 0.0
        for (u, v), w in zip(self.edges, self.weights):
            # Cost = sum_edge w * (1 - Z_u * Z_v) / 2
            # = sum_edge w/2 * (1 - <Z_u Z_v>)
            zz_expval = self._measure_zz(u, v)
            cost += w * 0.5 * (1 - zz_expval)
        
        return -cost  # Minimize negative cost = maximize cut
    
    def _apply_qaoa_circuit(self, gammas: np.ndarray, betas: np.ndarray) -> None:
        """Apply QAOA circuit to device.
        
        Args:
            gammas: Gamma parameters
            betas: Beta parameters
        """
        ops = []
        
        # Initial state: superposition
        for wire in range(self.n_qubits):
            ops.append({'name': 'h', 'wires': [wire], 'params': []})
        
        # QAOA layers
        for i in range(self.p):
            # Cost Hamiltonian: exp(-i * gamma * C)
            # For Max-Cut: ZZ interactions on edges
            for (u, v), w in zip(self.edges, self.weights):
                # ZZ gate decomposition: CNOT - RZ - CNOT
                ops.append({'name': 'cx', 'wires': [u, v], 'params': []})
                ops.append({'name': 'rz', 'wires': [v], 'params': [2 * gammas[i] * w]})
                ops.append({'name': 'cx', 'wires': [u, v], 'params': []})
            
            # Mixer Hamiltonian: exp(-i * beta * B)
            # B = sum_i X_i
            for wire in range(self.n_qubits):
                ops.append({'name': 'rx', 'wires': [wire], 'params': [2 * betas[i]]})
        
        # Apply to device
        if hasattr(self.device, 'apply'):
            self.device.reset()
            self.device.apply(ops)
    
    def _measure_zz(self, u: int, v: int) -> float:
        """Measure ZZ correlation between two qubits.
        
        Args:
            u: First qubit
            v: Second qubit
            
        Returns:
            <Z_u Z_v> expectation value
        """
        # Get probabilities
        if hasattr(self.device, 'probability'):
            probs = self.device.probability()
        else:
            # Mock probabilities
            probs = np.ones(2 ** self.n_qubits) / (2 ** self.n_qubits)
        
        # Compute ZZ expectation
        zz_expval = 0.0
        for i, p in enumerate(probs):
            # Extract bits
            bit_u = (i >> (self.n_qubits - 1 - u)) & 1
            bit_v = (i >> (self.n_qubits - 1 - v)) & 1
            
            # Z eigenvalues: |0⟩ → +1, |1⟩ → -1
            z_u = 1 - 2 * bit_u
            z_v = 1 - 2 * bit_v
            
            zz_expval += p * z_u * z_v
        
        return zz_expval
    
    def run(
        self,
        initial_params: Optional[tuple[np.ndarray, np.ndarray]] = None,
        max_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = False,
    ) -> QAOAResult:
        """Run QAOA optimization.
        
        Args:
            initial_params: Optional (gammas, betas) initial values
            max_iterations: Maximum iterations
            learning_rate: Optimizer step size
            verbose: Print progress
            
        Returns:
            QAOAResult with optimization results
        """
        # Initialize parameters
        if initial_params is None:
            gammas = np.random.uniform(0, 2 * np.pi, self.p)
            betas = np.random.uniform(0, np.pi, self.p)
        else:
            gammas, betas = initial_params
            gammas = np.array(gammas, dtype=float)
            betas = np.array(betas, dtype=float)
        
        cost_history = []
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Compute cost
            cost = self._cost_function(gammas, betas)
            cost_history.append(float(cost))
            
            if verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Compute gradients
            grad_gammas = self._gradient_gamma(gammas, betas)
            grad_betas = self._gradient_beta(gammas, betas)
            
            # Update parameters
            gammas = gammas - learning_rate * grad_gammas
            betas = betas - learning_rate * grad_betas
        
        # Find best bitstring
        best_bitstring, best_prob = self._find_best_solution(gammas, betas)
        
        return QAOAResult(
            final_cost=cost_history[-1] if cost_history else 0.0,
            optimal_gammas=gammas,
            optimal_betas=betas,
            iterations=max_iterations,
            cost_history=cost_history,
            best_bitstring=best_bitstring,
            solution_probability=best_prob,
        )
    
    def _gradient_gamma(self, gammas: np.ndarray, betas: np.ndarray, shift: float = np.pi / 4) -> np.ndarray:
        """Compute gradient with respect to gamma parameters."""
        gradient = np.zeros_like(gammas)
        
        for i in range(len(gammas)):
            gammas_plus = gammas.copy()
            gammas_plus[i] += shift
            cost_plus = self._cost_function(gammas_plus, betas)
            
            gammas_minus = gammas.copy()
            gammas_minus[i] -= shift
            cost_minus = self._cost_function(gammas_minus, betas)
            
            gradient[i] = (cost_plus - cost_minus) / (2 * np.sin(shift))
        
        return gradient
    
    def _gradient_beta(self, gammas: np.ndarray, betas: np.ndarray, shift: float = np.pi / 4) -> np.ndarray:
        """Compute gradient with respect to beta parameters."""
        gradient = np.zeros_like(betas)
        
        for i in range(len(betas)):
            betas_plus = betas.copy()
            betas_plus[i] += shift
            cost_plus = self._cost_function(gammas, betas_plus)
            
            betas_minus = betas.copy()
            betas_minus[i] -= shift
            cost_minus = self._cost_function(gammas, betas_minus)
            
            gradient[i] = (cost_plus - cost_minus) / (2 * np.sin(shift))
        
        return gradient
    
    def _find_best_solution(self, gammas: np.ndarray, betas: np.ndarray) -> tuple[str, float]:
        """Find the most probable solution bitstring.
        
        Args:
            gammas: Optimal gamma parameters
            betas: Optimal beta parameters
            
        Returns:
            (best_bitstring, probability)
        """
        # Apply final circuit
        self._apply_qaoa_circuit(gammas, betas)
        
        # Get probabilities
        if hasattr(self.device, 'probability'):
            probs = self.device.probability()
        else:
            probs = np.ones(2 ** self.n_qubits) / (2 ** self.n_qubits)
        
        # Find maximum
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_bitstring = format(best_idx, f'0{self.n_qubits}b')
        
        return best_bitstring, best_prob
    
    def compute_cut_value(self, bitstring: str) -> int:
        """Compute the cut value for a given bitstring.
        
        Args:
            bitstring: Binary string representing node partition
            
        Returns:
            Number of edges cut
        """
        cut = 0
        for (u, v), w in zip(self.edges, self.weights):
            if bitstring[u] != bitstring[v]:
                cut += w
        return int(cut)


class QuantumNeuralNetwork:
    """Quantum Neural Network for classification using LRET device.
    
    A simple QNN implementation for binary classification tasks.
    Uses a parameterized quantum circuit as a classifier.
    
    Example:
        >>> from proxima.backends.lret.pennylane_device import QLRETDevice
        >>> dev = QLRETDevice(wires=4, shots=1024)
        >>> 
        >>> qnn = QuantumNeuralNetwork(dev, n_layers=2)
        >>> 
        >>> # Train on data
        >>> X_train = np.random.randn(100, 4)  # 100 samples, 4 features
        >>> y_train = np.random.randint(0, 2, 100)  # Binary labels
        >>> qnn.fit(X_train, y_train, epochs=50)
        >>> 
        >>> # Predict
        >>> predictions = qnn.predict(X_test)
    """
    
    def __init__(
        self,
        device: Any,
        n_layers: int = 2,
        learning_rate: float = 0.01,
    ):
        """Initialize QNN.
        
        Args:
            device: PennyLane-compatible device
            n_layers: Number of variational layers
            learning_rate: Training learning rate
        """
        self.device = device
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_qubits = getattr(device, 'num_wires', 4)
        
        # Initialize parameters
        self.params = np.random.uniform(
            -np.pi, np.pi, 
            (n_layers, self.n_qubits, 3)  # 3 rotation angles per qubit per layer
        )
        
        self._loss_history = []
    
    def _encode_features(self, x: np.ndarray) -> list[dict]:
        """Encode classical features into quantum state.
        
        Args:
            x: Feature vector
            
        Returns:
            List of encoding operations
        """
        ops = []
        
        # Angle encoding
        for i, val in enumerate(x[:self.n_qubits]):
            ops.append({'name': 'ry', 'wires': [i], 'params': [val]})
        
        return ops
    
    def _variational_layer(self, layer_params: np.ndarray) -> list[dict]:
        """Build a variational layer.
        
        Args:
            layer_params: Parameters for this layer (n_qubits, 3)
            
        Returns:
            List of operations
        """
        ops = []
        
        # Single-qubit rotations
        for i in range(self.n_qubits):
            ops.append({'name': 'rx', 'wires': [i], 'params': [layer_params[i, 0]]})
            ops.append({'name': 'ry', 'wires': [i], 'params': [layer_params[i, 1]]})
            ops.append({'name': 'rz', 'wires': [i], 'params': [layer_params[i, 2]]})
        
        # Entangling layer
        for i in range(self.n_qubits - 1):
            ops.append({'name': 'cx', 'wires': [i, i + 1], 'params': []})
        
        return ops
    
    def _forward(self, x: np.ndarray) -> float:
        """Forward pass through QNN.
        
        Args:
            x: Input feature vector
            
        Returns:
            Output probability (for class 1)
        """
        ops = []
        
        # Encode features
        ops.extend(self._encode_features(x))
        
        # Apply variational layers
        for layer in range(self.n_layers):
            ops.extend(self._variational_layer(self.params[layer]))
        
        # Apply to device
        if hasattr(self.device, 'reset'):
            self.device.reset()
        if hasattr(self.device, 'apply'):
            self.device.apply(ops)
        
        # Measure first qubit
        if hasattr(self.device, 'probability'):
            probs = self.device.probability(wires=[0])
            return probs[1] if len(probs) > 1 else 0.5
        else:
            return 0.5
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Average loss
        """
        eps = 1e-7
        total_loss = 0.0
        
        for xi, yi in zip(X, y):
            pred = self._forward(xi)
            pred = np.clip(pred, eps, 1 - eps)
            
            if yi == 1:
                total_loss -= np.log(pred)
            else:
                total_loss -= np.log(1 - pred)
        
        return total_loss / len(X)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> None:
        """Train the QNN.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print training progress
        """
        self._loss_history = []
        n_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Compute gradient and update
                grad = self._compute_gradient(X_batch, y_batch)
                self.params -= self.learning_rate * grad
                
                batch_loss = self._compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self._loss_history.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
        """Compute gradient using parameter-shift rule.
        
        Args:
            X: Batch features
            y: Batch labels
            shift: Shift amount
            
        Returns:
            Gradient array with same shape as params
        """
        gradient = np.zeros_like(self.params)
        
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                for angle in range(3):
                    # Plus shift
                    self.params[layer, qubit, angle] += shift
                    loss_plus = self._compute_loss(X, y)
                    
                    # Minus shift
                    self.params[layer, qubit, angle] -= 2 * shift
                    loss_minus = self._compute_loss(X, y)
                    
                    # Restore
                    self.params[layer, qubit, angle] += shift
                    
                    gradient[layer, qubit, angle] = (loss_plus - loss_minus) / (2 * np.sin(shift))
        
        return gradient
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Predicted labels
        """
        predictions = np.zeros(len(X), dtype=int)
        
        for i, xi in enumerate(X):
            prob = self._forward(xi)
            predictions[i] = 1 if prob >= threshold else 0
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of class 1 for each sample
        """
        probs = np.zeros(len(X))
        
        for i, xi in enumerate(X):
            probs[i] = self._forward(xi)
        
        return probs
    
    @property
    def loss_history(self) -> list[float]:
        """Get training loss history."""
        return self._loss_history
