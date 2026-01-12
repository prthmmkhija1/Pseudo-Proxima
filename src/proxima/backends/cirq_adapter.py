"""Cirq backend adapter (DensityMatrix + StateVector) with comprehensive error handling."""

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
from proxima.backends.exceptions import (
    BackendNotInstalledError,
    CircuitValidationError,
    QubitLimitExceededError,
    wrap_backend_exception,
)


class CirqBackendAdapter(BaseBackendAdapter):
    def get_name(self) -> str:
        return "cirq"

    def get_version(self) -> str:
        spec = importlib.util.find_spec("cirq")
        if spec and spec.loader:
            try:
                import cirq

                return getattr(cirq, "__version__", "unknown")
            except Exception:
                return "unknown"
        return "unavailable"

    def is_available(self) -> bool:
        return importlib.util.find_spec("cirq") is not None

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=30,
            supports_noise=True,
            supports_gpu=False,
            supports_batching=False,
            custom_features={},
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        if not self.is_available():
            return ValidationResult(valid=False, message="cirq not installed")
        try:
            import cirq
        except Exception as exc:  # pragma: no cover - defensive
            return ValidationResult(valid=False, message=f"cirq import failed: {exc}")

        if not isinstance(circuit, getattr(cirq, "Circuit", ())):
            return ValidationResult(valid=False, message="input is not a cirq.Circuit")
        return ValidationResult(valid=True, message="ok")

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "cirq not installed"}
            )
        try:
            import cirq
        except Exception:
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "cirq import failed"}
            )

        if not isinstance(circuit, getattr(cirq, "Circuit", ())):
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "not a cirq.Circuit"}
            )

        qubits = len(circuit.all_qubits())
        gate_count = sum(len(m) for m in circuit)
        depth = len(circuit)  # Number of moments = depth
        # Estimate memory: statevector needs 2^n * 16 bytes (complex128)
        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 30 else None
        metadata = {
            "qubits": qubits,
            "gate_count": gate_count,
            "depth": depth,
        }
        return ResourceEstimate(memory_mb=memory_mb, time_ms=None, metadata=metadata)

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        if not self.is_available():
            raise BackendNotInstalledError("cirq", ["cirq"])

        # Validate circuit first
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(
                backend_name="cirq",
                reason=validation.message or "Invalid circuit",
            )

        try:
            import cirq
        except ImportError as exc:
            raise BackendNotInstalledError("cirq", ["cirq"], original_exception=exc)

        options = options or {}
        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        repetitions = int(options.get("repetitions", options.get("shots", 0)))

        # Check qubit limits
        qubit_count = len(circuit.all_qubits()) if hasattr(circuit, "all_qubits") else 0
        max_qubits = self.get_capabilities().max_qubits
        if qubit_count > max_qubits:
            raise QubitLimitExceededError(
                backend_name="cirq",
                requested_qubits=qubit_count,
                max_qubits=max_qubits,
            )

        if sim_type not in (SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX):
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        try:
            simulator: Any
            if sim_type == SimulatorType.DENSITY_MATRIX:
                simulator = cirq.DensityMatrixSimulator()
            else:
                simulator = cirq.Simulator()

            start = time.perf_counter()
            result_type: ResultType
            data: dict[str, Any]
            raw_result: Any

            if repetitions > 0:
                raw_result = simulator.run(circuit, repetitions=repetitions)
                result_type = ResultType.COUNTS
                counts: dict[str, int] = {}
                measurement_keys = list(raw_result.measurements.keys())
                if measurement_keys:
                    for key in measurement_keys:
                        histogram = raw_result.histogram(key=key)
                        for state_int, count in histogram.items():
                            n_bits = raw_result.measurements[key].shape[1]
                            bitstring = format(state_int, f"0{n_bits}b")
                            counts[bitstring] = counts.get(bitstring, 0) + count
                data = {"counts": counts, "repetitions": repetitions}
            else:
                if sim_type == SimulatorType.DENSITY_MATRIX:
                    raw_result = simulator.simulate(circuit)
                    density_matrix = raw_result.final_density_matrix
                    result_type = ResultType.DENSITY_MATRIX
                    data = {"density_matrix": density_matrix}
                else:
                    raw_result = simulator.simulate(circuit)
                    statevector = raw_result.final_state_vector
                    result_type = ResultType.STATEVECTOR
                    data = {"statevector": statevector}

            execution_time_ms = (time.perf_counter() - start) * 1000.0

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=sim_type,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=repetitions if repetitions > 0 else None,
                result_type=result_type,
                data=data,
                metadata={"cirq_version": self.get_version()},
                raw_result=raw_result,
            )

        except (
            BackendNotInstalledError,
            CircuitValidationError,
            QubitLimitExceededError,
        ):
            raise
        except Exception as exc:
            raise wrap_backend_exception(exc, "cirq", "execution")

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        return sim_type in self.get_capabilities().simulator_types
