"""LRET backend adapter with comprehensive implementation.

LRET (Lightweight Runtime Execution Toolkit) integration for quantum simulations.
Target: https://github.com/kunal5556/LRET (feature/framework-integration branch)
"""

from __future__ import annotations

import importlib.util
import logging
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
    UnsupportedOperationError,
    wrap_backend_exception,
)


class LRETBackendAdapter(BaseBackendAdapter):
    """LRET backend adapter for quantum circuit simulation.

    LRET is a lightweight framework for quantum computing experiments.
    This adapter supports the feature/framework-integration branch API.
    """

    def __init__(self) -> None:
        """Initialize the LRET adapter."""
        self._lret_module: Any = None
        self._cached_version: str | None = None

    def get_name(self) -> str:
        """Return backend identifier."""
        return "lret"

    def get_version(self) -> str:
        """Return LRET version string."""
        if self._cached_version:
            return self._cached_version

        if not self.is_available():
            return "unavailable"

        try:
            lret = self._get_lret_module()
            self._cached_version = getattr(lret, "__version__", "unknown")
            return self._cached_version
        except Exception:
            return "unknown"

    def is_available(self) -> bool:
        """Check if LRET is installed and importable."""
        return importlib.util.find_spec("lret") is not None

    def _get_lret_module(self) -> Any:
        """Get the LRET module, importing if needed."""
        if self._lret_module is None:
            if not self.is_available():
                raise BackendNotInstalledError("lret", ["lret"])
            import lret  # type: ignore

            self._lret_module = lret
        return self._lret_module

    def get_capabilities(self) -> Capabilities:
        """Return LRET capabilities.

        LRET supports custom simulation modes that may differ from
        standard StateVector/DensityMatrix simulators.
        """
        return Capabilities(
            simulator_types=[SimulatorType.CUSTOM, SimulatorType.STATE_VECTOR],
            max_qubits=32,
            supports_noise=False,
            supports_gpu=False,
            supports_batching=True,
            custom_features={
                "framework_integration": True,
                "custom_gates": True,
                "lret_native_format": True,
            },
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate a circuit for LRET execution.

        LRET accepts various circuit formats:
        - Native LRET circuit objects
        - Dictionary-based circuit specifications
        - Qiskit/Cirq circuits (via conversion)

        Args:
            circuit: Circuit to validate

        Returns:
            ValidationResult indicating validity
        """
        if circuit is None:
            return ValidationResult(
                valid=False,
                message="Circuit is None",
                details={"error": "null_circuit"},
            )

        # Check for LRET native format
        if self.is_available():
            try:
                lret = self._get_lret_module()

                # Check for native LRET circuit type
                if hasattr(lret, "Circuit"):
                    if isinstance(circuit, lret.Circuit):
                        return ValidationResult(valid=True, message="LRET native circuit")

                # Check for LRET-compatible dict format
                if hasattr(lret, "validate_circuit"):
                    is_valid = lret.validate_circuit(circuit)
                    return ValidationResult(
                        valid=is_valid,
                        message="Validated via LRET" if is_valid else "LRET validation failed",
                    )
            except Exception as exc:
                logging.debug("LRET validation check failed: %s", exc)

        # Generic validation for dict-based circuits
        if isinstance(circuit, dict):
            required_keys = {"gates"} | {"operations"} | {"instructions"}
            has_required = bool(required_keys & set(circuit.keys()))
            if has_required or "qubits" in circuit:
                return ValidationResult(
                    valid=True,
                    message="Dictionary-based circuit accepted",
                    details={"format": "dict", "keys": list(circuit.keys())},
                )
            return ValidationResult(
                valid=False,
                message="Dictionary circuit missing required keys",
                details={"provided_keys": list(circuit.keys())},
            )

        # Check for Qiskit/Cirq circuits that can be converted
        circuit_type = type(circuit).__name__
        if "QuantumCircuit" in circuit_type or "Circuit" in circuit_type:
            return ValidationResult(
                valid=True,
                message=f"Circuit type {circuit_type} will be converted for LRET",
                details={"original_type": circuit_type, "requires_conversion": True},
            )

        return ValidationResult(
            valid=False,
            message=f"Unsupported circuit type: {circuit_type}",
            details={"type": circuit_type},
        )

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate resources needed for circuit execution."""
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "LRET not installed"},
            )

        qubits = self._extract_qubit_count(circuit)
        if qubits is None:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "Could not determine qubit count"},
            )

        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 32 else None
        gate_count = self._extract_gate_count(circuit)

        metadata: dict[str, Any] = {"qubits": qubits}
        if gate_count is not None:
            metadata["gate_count"] = gate_count
            time_ms = gate_count * qubits * 0.1
        else:
            time_ms = None

        return ResourceEstimate(memory_mb=memory_mb, time_ms=time_ms, metadata=metadata)

    def _extract_qubit_count(self, circuit: Any) -> int | None:
        """Extract qubit count from various circuit formats."""
        for attr in ("num_qubits", "n_qubits", "qubits", "qubit_count"):
            val = getattr(circuit, attr, None)
            if isinstance(val, int):
                return val
            if hasattr(val, "__len__"):
                return len(val)

        if isinstance(circuit, dict):
            for key in ("num_qubits", "qubits", "n_qubits"):
                val = circuit.get(key)
                if isinstance(val, int):
                    return val
                if isinstance(val, list):
                    return len(val)

        if hasattr(circuit, "all_qubits"):
            try:
                return len(circuit.all_qubits())
            except Exception as exc:
                logging.debug("Could not extract qubit count via all_qubits(): %s", exc)

        return None

    def _extract_gate_count(self, circuit: Any) -> int | None:
        """Extract gate count from various circuit formats."""
        if hasattr(circuit, "size"):
            try:
                return circuit.size()
            except Exception as exc:
                logging.debug("Could not extract gate count via size(): %s", exc)

        if isinstance(circuit, dict):
            for key in ("gates", "operations", "instructions"):
                val = circuit.get(key)
                if isinstance(val, list):
                    return len(val)

        return None

    def execute(self, circuit: Any, options: dict[str, Any] | None = None) -> ExecutionResult:
        """Execute a circuit using LRET.

        Args:
            circuit: The circuit to execute
            options: Execution options

        Returns:
            ExecutionResult with simulation results
        """
        if not self.is_available():
            raise BackendNotInstalledError("lret", ["lret"])

        options = options or {}
        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        shots = int(options.get("shots", options.get("repetitions", 0)))
        seed = options.get("seed")

        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(
                backend_name="lret",
                reason=validation.message or "Validation failed",
                circuit_info=validation.details,
            )

        try:
            lret = self._get_lret_module()
            start_time = time.perf_counter()

            lret_circuit = self._prepare_circuit(circuit, lret)
            result_data: dict[str, Any] = {}
            result_type: ResultType
            raw_result: Any = None

            if hasattr(lret, "Simulator"):
                simulator = lret.Simulator()
                if seed is not None and hasattr(simulator, "set_seed"):
                    simulator.set_seed(seed)

                if shots > 0:
                    raw_result = simulator.run(lret_circuit, shots=shots)
                    counts = self._extract_counts(raw_result)
                    result_data = {"counts": counts, "shots": shots}
                    result_type = ResultType.COUNTS
                else:
                    raw_result = simulator.simulate(lret_circuit)
                    statevector = self._extract_statevector(raw_result)
                    result_data = {"statevector": statevector}
                    result_type = ResultType.STATEVECTOR

            elif hasattr(lret, "execute"):
                raw_result = lret.execute(lret_circuit, shots=shots if shots > 0 else None)
                if shots > 0:
                    counts = self._extract_counts(raw_result)
                    result_data = {"counts": counts, "shots": shots}
                    result_type = ResultType.COUNTS
                else:
                    statevector = self._extract_statevector(raw_result)
                    result_data = {"statevector": statevector}
                    result_type = ResultType.STATEVECTOR
            else:
                raise UnsupportedOperationError(
                    backend_name="lret",
                    operation="execute",
                    supported_operations=["Simulator.run", "execute"],
                )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            qubit_count = self._extract_qubit_count(circuit) or 0

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=sim_type,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=shots if shots > 0 else None,
                result_type=result_type,
                data=result_data,
                metadata={"lret_version": self.get_version(), "seed": seed},
                raw_result=raw_result,
            )

        except (BackendNotInstalledError, CircuitValidationError, UnsupportedOperationError):
            raise
        except Exception as exc:
            raise wrap_backend_exception(exc, "lret", "execution")

    def _prepare_circuit(self, circuit: Any, lret: Any) -> Any:
        """Prepare/convert circuit for LRET execution."""
        if hasattr(lret, "Circuit") and isinstance(circuit, lret.Circuit):
            return circuit
        if hasattr(lret, "from_qiskit") and "QuantumCircuit" in type(circuit).__name__:
            return lret.from_qiskit(circuit)
        if hasattr(lret, "from_cirq") and "cirq" in type(circuit).__module__.lower():
            return lret.from_cirq(circuit)
        if hasattr(lret, "from_dict") and isinstance(circuit, dict):
            return lret.from_dict(circuit)
        return circuit

    def _extract_counts(self, result: Any) -> dict[str, int]:
        """Extract measurement counts from LRET result."""
        for attr in ("counts", "get_counts", "measurements"):
            val = getattr(result, attr, None)
            if isinstance(val, dict):
                return val
            if callable(val):
                try:
                    return val()
                except Exception as exc:
                    logging.debug("Could not extract counts via %s(): %s", attr, exc)
        return {}

    def _extract_statevector(self, result: Any) -> Any:
        """Extract statevector from LRET result."""
        import numpy as np

        for attr in ("statevector", "state_vector", "final_state", "state"):
            val = getattr(result, attr, None)
            if val is not None:
                return np.asarray(val, dtype=complex)

        if hasattr(result, "__array__"):
            return np.asarray(result, dtype=complex)

        return None

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Check if simulator type is supported."""
        return sim_type in self.get_capabilities().simulator_types

    def get_supported_gates(self) -> list[str]:
        """Get list of gates supported by LRET."""
        standard_gates = [
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "T",
            "Sdg",
            "Tdg",
            "RX",
            "RY",
            "RZ",
            "CX",
            "CNOT",
            "CZ",
            "SWAP",
            "CCX",
            "U",
            "U1",
            "U2",
            "U3",
        ]
        if self.is_available():
            try:
                lret = self._get_lret_module()
                if hasattr(lret, "SUPPORTED_GATES"):
                    return list(lret.SUPPORTED_GATES)
            except Exception as exc:
                logging.debug("Could not get LRET supported gates: %s", exc)
        return standard_gates
