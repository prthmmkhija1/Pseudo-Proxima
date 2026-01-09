"""Qiskit backend adapter for Statevector and DensityMatrix simulation."""

from __future__ import annotations

import importlib.util
import time
from typing import Any

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)


class QiskitBackendAdapter(BaseBackendAdapter):
    def get_name(self) -> str:
        return "qiskit"

    def get_version(self) -> str:
        spec = importlib.util.find_spec("qiskit")
        if spec and spec.loader:
            try:
                import qiskit

                return getattr(qiskit, "__version__", "unknown")
            except Exception:
                return "unknown"
        return "unavailable"

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("qiskit") is not None
            and importlib.util.find_spec("qiskit_aer") is not None
        )

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=28,
            supports_noise=True,
            supports_gpu=False,
            supports_batching=False,
            custom_features={"transpilation": True},
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        if not self.is_available():
            return ValidationResult(valid=False, message="qiskit/qiskit-aer not installed")
        try:
            from qiskit import QuantumCircuit
        except Exception as exc:  # pragma: no cover - defensive
            return ValidationResult(valid=False, message=f"qiskit import failed: {exc}")

        if not isinstance(circuit, QuantumCircuit):
            return ValidationResult(valid=False, message="input is not a qiskit.QuantumCircuit")
        return ValidationResult(valid=True, message="ok")

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "qiskit/qiskit-aer not installed"}
            )
        try:
            from qiskit import QuantumCircuit
        except Exception:
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "qiskit import failed"}
            )

        if not isinstance(circuit, QuantumCircuit):
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "not a QuantumCircuit"}
            )

        qubits = circuit.num_qubits
        depth = circuit.depth() or 0
        gate_count = circuit.size()
        # Estimate memory: statevector needs 2^n * 16 bytes (complex128)
        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 28 else None
        metadata = {
            "qubits": qubits,
            "depth": depth,
            "gate_count": gate_count,
        }
        return ResourceEstimate(memory_mb=memory_mb, time_ms=None, metadata=metadata)

    def execute(self, circuit: Any, options: dict[str, Any] | None = None) -> ExecutionResult:
        if not self.is_available():
            raise RuntimeError("qiskit/qiskit-aer is not installed")
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Expected qiskit.QuantumCircuit")

        options = options or {}
        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        shots = int(options.get("shots", options.get("repetitions", 0)))
        density_mode = sim_type == SimulatorType.DENSITY_MATRIX

        if sim_type not in (SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX):
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        method = "density_matrix" if density_mode else "statevector"
        simulator = AerSimulator(method=method)
        t_circuit = transpile(circuit, simulator)

        # Clone circuit to add save instruction for non-shot execution
        exec_circuit = t_circuit.copy()
        if shots == 0:
            if density_mode:
                exec_circuit.save_density_matrix()
            else:
                exec_circuit.save_statevector()

        start = time.perf_counter()
        result = simulator.run(exec_circuit, shots=shots if shots > 0 else None).result()
        execution_time_ms = (time.perf_counter() - start) * 1000.0

        if shots > 0:
            counts = result.get_counts(exec_circuit)
            data = {"counts": counts, "shots": shots}
            result_type = ResultType.COUNTS
            raw_result = result
        else:
            result_data = result.data(exec_circuit)
            if density_mode:
                density_matrix = result_data.get("density_matrix")
                result_type = ResultType.DENSITY_MATRIX
                data = {"density_matrix": density_matrix}
                raw_result = result
            else:
                statevector = result_data.get("statevector")
                result_type = ResultType.STATEVECTOR
                data = {"statevector": statevector}
                raw_result = result

        qubit_count = circuit.num_qubits
        return ExecutionResult(
            backend=self.get_name(),
            simulator_type=sim_type,
            execution_time_ms=execution_time_ms,
            qubit_count=qubit_count,
            shot_count=shots if shots > 0 else None,
            result_type=result_type,
            data=data,
            metadata={},
            raw_result=raw_result,
        )

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        return sim_type in self.get_capabilities().simulator_types
