"""Circuit conversion utilities for cross-backend interoperability."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitFormat(str, Enum):
    """Supported circuit formats."""

    QISKIT = "qiskit"
    CIRQ = "cirq"
    OPENQASM2 = "openqasm2"
    OPENQASM3 = "openqasm3"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass
class CircuitInfo:
    """Information extracted from a circuit."""

    format: CircuitFormat
    num_qubits: int
    num_classical_bits: int = 0
    depth: int = 0
    gate_count: int = 0
    gate_types: dict[str, int] = field(default_factory=dict)
    has_measurements: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionResult:
    """Result of a circuit conversion."""

    success: bool
    circuit: Any = None
    source_format: CircuitFormat = CircuitFormat.UNKNOWN
    target_format: CircuitFormat = CircuitFormat.UNKNOWN
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def detect_circuit_format(circuit: Any) -> CircuitFormat:
    """Detect the format/library of a circuit object.

    Args:
        circuit: A circuit object from any supported library

    Returns:
        CircuitFormat indicating the detected format
    """
    if circuit is None:
        return CircuitFormat.UNKNOWN

    type_name = type(circuit).__module__ + "." + type(circuit).__name__

    # Check for Qiskit
    if "qiskit" in type_name.lower():
        return CircuitFormat.QISKIT

    # Check for Cirq
    if "cirq" in type_name.lower():
        return CircuitFormat.CIRQ

    # Check for string formats
    if isinstance(circuit, str):
        circuit_str = circuit.strip()
        if circuit_str.startswith("OPENQASM 3"):
            return CircuitFormat.OPENQASM3
        if circuit_str.startswith("OPENQASM 2") or circuit_str.startswith("OPENQASM"):
            return CircuitFormat.OPENQASM2
        if circuit_str.startswith("{"):
            return CircuitFormat.JSON

    return CircuitFormat.UNKNOWN


def extract_circuit_info(circuit: Any) -> CircuitInfo:
    """Extract information from a circuit regardless of format.

    Args:
        circuit: A circuit object from any supported library

    Returns:
        CircuitInfo with extracted details
    """
    fmt = detect_circuit_format(circuit)

    if fmt == CircuitFormat.QISKIT:
        return _extract_qiskit_info(circuit)
    elif fmt == CircuitFormat.CIRQ:
        return _extract_cirq_info(circuit)
    elif fmt in (CircuitFormat.OPENQASM2, CircuitFormat.OPENQASM3):
        return _extract_qasm_info(circuit, fmt)

    # Unknown format - try to extract basic info
    return CircuitInfo(
        format=fmt,
        num_qubits=getattr(circuit, "num_qubits", 0),
        metadata={"raw_type": type(circuit).__name__},
    )


def _extract_qiskit_info(circuit: Any) -> CircuitInfo:
    """Extract info from a Qiskit QuantumCircuit."""
    gate_types: dict[str, int] = {}

    try:
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
    except Exception as exc:
        logging.debug("Could not extract Qiskit gate types: %s", exc)

    has_measurements = "measure" in gate_types

    return CircuitInfo(
        format=CircuitFormat.QISKIT,
        num_qubits=circuit.num_qubits,
        num_classical_bits=circuit.num_clbits,
        depth=circuit.depth() if hasattr(circuit, "depth") else 0,
        gate_count=circuit.size() if hasattr(circuit, "size") else len(gate_types),
        gate_types=gate_types,
        has_measurements=has_measurements,
        metadata={
            "name": getattr(circuit, "name", ""),
            "global_phase": float(getattr(circuit, "global_phase", 0)),
        },
    )


def _extract_cirq_info(circuit: Any) -> CircuitInfo:
    """Extract info from a Cirq Circuit."""
    gate_types: dict[str, int] = {}
    has_measurements = False

    try:
        for moment in circuit:
            for op in moment:
                gate_name = type(op.gate).__name__
                gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
                if "measure" in gate_name.lower():
                    has_measurements = True
    except Exception as exc:
        logging.debug("Could not extract Cirq gate types: %s", exc)

    num_qubits = len(circuit.all_qubits()) if hasattr(circuit, "all_qubits") else 0

    return CircuitInfo(
        format=CircuitFormat.CIRQ,
        num_qubits=num_qubits,
        num_classical_bits=0,  # Cirq handles this differently
        depth=len(circuit) if hasattr(circuit, "__len__") else 0,
        gate_count=sum(gate_types.values()),
        gate_types=gate_types,
        has_measurements=has_measurements,
        metadata={},
    )


def _extract_qasm_info(circuit_str: str, fmt: CircuitFormat) -> CircuitInfo:
    """Extract info from OpenQASM string."""
    lines = circuit_str.strip().split("\n")
    num_qubits = 0
    num_cbits = 0
    gate_types: dict[str, int] = {}

    for line in lines:
        line = line.strip()
        if line.startswith("qreg"):
            # Parse qreg q[N];
            try:
                parts = line.split("[")
                if len(parts) > 1:
                    num_qubits += int(parts[1].split("]")[0])
            except (ValueError, IndexError) as exc:
                logging.debug("Could not parse qreg line '%s': %s", line, exc)
        elif line.startswith("creg"):
            try:
                parts = line.split("[")
                if len(parts) > 1:
                    num_cbits += int(parts[1].split("]")[0])
            except (ValueError, IndexError) as exc:
                logging.debug("Could not parse creg line '%s': %s", line, exc)
        elif line and not line.startswith(("//", "OPENQASM", "include")):
            # Count gates
            gate = line.split()[0].split("(")[0]
            if gate:
                gate_types[gate] = gate_types.get(gate, 0) + 1

    return CircuitInfo(
        format=fmt,
        num_qubits=num_qubits,
        num_classical_bits=num_cbits,
        gate_count=sum(gate_types.values()),
        gate_types=gate_types,
        has_measurements="measure" in gate_types,
        metadata={"line_count": len(lines)},
    )


# ==============================================================================
# Circuit Conversion Functions
# ==============================================================================


def convert_circuit(
    circuit: Any,
    target_format: CircuitFormat,
    *,
    preserve_measurements: bool = True,
) -> ConversionResult:
    """Convert a circuit to a different format.

    Args:
        circuit: Source circuit in any supported format
        target_format: Desired output format
        preserve_measurements: Whether to keep measurement operations

    Returns:
        ConversionResult with converted circuit or error
    """
    source_format = detect_circuit_format(circuit)

    if source_format == CircuitFormat.UNKNOWN:
        return ConversionResult(
            success=False,
            source_format=source_format,
            target_format=target_format,
            error=f"Unknown circuit format: {type(circuit).__name__}",
        )

    if source_format == target_format:
        return ConversionResult(
            success=True,
            circuit=circuit,
            source_format=source_format,
            target_format=target_format,
            warnings=["Source and target formats are the same; no conversion needed"],
        )

    # Route to specific conversion functions
    converters = {
        (CircuitFormat.QISKIT, CircuitFormat.CIRQ): _qiskit_to_cirq,
        (CircuitFormat.CIRQ, CircuitFormat.QISKIT): _cirq_to_qiskit,
        (CircuitFormat.QISKIT, CircuitFormat.OPENQASM2): _qiskit_to_qasm2,
        (CircuitFormat.OPENQASM2, CircuitFormat.QISKIT): _qasm2_to_qiskit,
        (CircuitFormat.CIRQ, CircuitFormat.OPENQASM2): _cirq_to_qasm2,
        (CircuitFormat.OPENQASM2, CircuitFormat.CIRQ): _qasm2_to_cirq,
        (CircuitFormat.OPENQASM3, CircuitFormat.QISKIT): _qasm3_to_qiskit,
        (CircuitFormat.OPENQASM3, CircuitFormat.CIRQ): _qasm3_to_cirq,
    }

    converter = converters.get((source_format, target_format))
    if converter is None:
        return ConversionResult(
            success=False,
            source_format=source_format,
            target_format=target_format,
            error=f"Conversion from {source_format.value} to {target_format.value} not supported",
        )

    try:
        return converter(circuit, preserve_measurements)
    except Exception as exc:
        return ConversionResult(
            success=False,
            source_format=source_format,
            target_format=target_format,
            error=f"Conversion failed: {exc}",
        )


def _qiskit_to_cirq(circuit: Any, preserve_measurements: bool) -> ConversionResult:
    """Convert Qiskit circuit to Cirq."""
    warnings: list[str] = []

    # Check if cirq is available
    if importlib.util.find_spec("cirq") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.QISKIT,
            target_format=CircuitFormat.CIRQ,
            error="Cirq is not installed",
        )

    import cirq

    # Create Cirq qubits
    n_qubits = circuit.num_qubits
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]

    # Map Qiskit gates to Cirq
    gate_map = {
        "h": cirq.H,
        "x": cirq.X,
        "y": cirq.Y,
        "z": cirq.Z,
        "s": cirq.S,
        "t": cirq.T,
        "cx": cirq.CNOT,
        "cz": cirq.CZ,
        "swap": cirq.SWAP,
        "id": cirq.I,
        "sdg": lambda: cirq.S**-1,
        "tdg": lambda: cirq.T**-1,
    }

    operations: list[Any] = []

    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        qubit_indices = [circuit.find_bit(q).index for q in instruction.qubits]
        target_qubits = [qubits[i] for i in qubit_indices]

        if gate_name == "measure":
            if preserve_measurements:
                operations.append(cirq.measure(*target_qubits))
            continue

        if gate_name == "barrier":
            warnings.append("Barrier operations are ignored in Cirq")
            continue

        if gate_name in gate_map:
            gate = gate_map[gate_name]
            if callable(gate) and not isinstance(gate, type):
                gate = gate()
            operations.append(gate.on(*target_qubits))
        elif gate_name == "rx":
            theta = float(instruction.operation.params[0])
            operations.append(cirq.rx(theta).on(*target_qubits))
        elif gate_name == "ry":
            theta = float(instruction.operation.params[0])
            operations.append(cirq.ry(theta).on(*target_qubits))
        elif gate_name == "rz":
            theta = float(instruction.operation.params[0])
            operations.append(cirq.rz(theta).on(*target_qubits))
        else:
            warnings.append(f"Gate '{gate_name}' not directly mapped; skipped")

    cirq_circuit = cirq.Circuit(operations)

    return ConversionResult(
        success=True,
        circuit=cirq_circuit,
        source_format=CircuitFormat.QISKIT,
        target_format=CircuitFormat.CIRQ,
        warnings=warnings,
    )


def _cirq_to_qiskit(circuit: Any, preserve_measurements: bool) -> ConversionResult:
    """Convert Cirq circuit to Qiskit."""
    warnings: list[str] = []

    if importlib.util.find_spec("qiskit") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.CIRQ,
            target_format=CircuitFormat.QISKIT,
            error="Qiskit is not installed",
        )

    import cirq
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    # Get all qubits and create mapping
    cirq_qubits = sorted(circuit.all_qubits())
    qubit_map = {q: i for i, q in enumerate(cirq_qubits)}
    n_qubits = len(cirq_qubits)

    # Create Qiskit circuit
    qr = QuantumRegister(n_qubits, "q")
    has_measurements = any(
        isinstance(op.gate, cirq.MeasurementGate) for moment in circuit for op in moment
    )
    cr = ClassicalRegister(n_qubits, "c") if has_measurements and preserve_measurements else None

    qc = QuantumCircuit(qr, cr) if cr else QuantumCircuit(qr)

    # Map Cirq gates to Qiskit
    for moment in circuit:
        for op in moment:
            gate_name = type(op.gate).__name__
            qubit_indices = [qubit_map[q] for q in op.qubits]

            if isinstance(op.gate, cirq.MeasurementGate):
                if preserve_measurements and cr:
                    for _i, qi in enumerate(qubit_indices):
                        qc.measure(qr[qi], cr[qi])
                continue

            # Handle common gates
            if isinstance(op.gate, cirq.HPowGate) and op.gate.exponent == 1:
                qc.h(qr[qubit_indices[0]])
            elif isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1:
                qc.x(qr[qubit_indices[0]])
            elif isinstance(op.gate, cirq.YPowGate) and op.gate.exponent == 1:
                qc.y(qr[qubit_indices[0]])
            elif isinstance(op.gate, cirq.ZPowGate) and op.gate.exponent == 1:
                qc.z(qr[qubit_indices[0]])
            elif isinstance(op.gate, cirq.CNotPowGate) and op.gate.exponent == 1:
                qc.cx(qr[qubit_indices[0]], qr[qubit_indices[1]])
            elif isinstance(op.gate, cirq.CZPowGate) and op.gate.exponent == 1:
                qc.cz(qr[qubit_indices[0]], qr[qubit_indices[1]])
            elif isinstance(op.gate, cirq.SwapPowGate) and op.gate.exponent == 1:
                qc.swap(qr[qubit_indices[0]], qr[qubit_indices[1]])
            elif hasattr(op.gate, "exponent"):
                # Handle parameterized gates
                exp = op.gate.exponent
                if isinstance(op.gate, cirq.XPowGate):
                    qc.rx(exp * 3.14159265359, qr[qubit_indices[0]])
                elif isinstance(op.gate, cirq.YPowGate):
                    qc.ry(exp * 3.14159265359, qr[qubit_indices[0]])
                elif isinstance(op.gate, cirq.ZPowGate):
                    qc.rz(exp * 3.14159265359, qr[qubit_indices[0]])
                else:
                    warnings.append(f"Gate '{gate_name}' not directly mapped; skipped")
            else:
                warnings.append(f"Gate '{gate_name}' not directly mapped; skipped")

    return ConversionResult(
        success=True,
        circuit=qc,
        source_format=CircuitFormat.CIRQ,
        target_format=CircuitFormat.QISKIT,
        warnings=warnings,
    )


def _qiskit_to_qasm2(circuit: Any, preserve_measurements: bool) -> ConversionResult:
    """Convert Qiskit circuit to OpenQASM 2.0."""
    try:
        qasm_str = circuit.qasm()
        return ConversionResult(
            success=True,
            circuit=qasm_str,
            source_format=CircuitFormat.QISKIT,
            target_format=CircuitFormat.OPENQASM2,
        )
    except Exception as exc:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.QISKIT,
            target_format=CircuitFormat.OPENQASM2,
            error=f"QASM export failed: {exc}",
        )


def _qasm2_to_qiskit(qasm_str: str, preserve_measurements: bool) -> ConversionResult:
    """Convert OpenQASM 2.0 to Qiskit circuit."""
    if importlib.util.find_spec("qiskit") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM2,
            target_format=CircuitFormat.QISKIT,
            error="Qiskit is not installed",
        )

    try:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit.from_qasm_str(qasm_str)
        return ConversionResult(
            success=True,
            circuit=circuit,
            source_format=CircuitFormat.OPENQASM2,
            target_format=CircuitFormat.QISKIT,
        )
    except Exception as exc:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM2,
            target_format=CircuitFormat.QISKIT,
            error=f"QASM import failed: {exc}",
        )


def _cirq_to_qasm2(circuit: Any, preserve_measurements: bool) -> ConversionResult:
    """Convert Cirq circuit to OpenQASM 2.0."""
    if importlib.util.find_spec("cirq") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.CIRQ,
            target_format=CircuitFormat.OPENQASM2,
            error="Cirq is not installed",
        )

    try:
        import cirq

        qasm_str = cirq.qasm(circuit)
        return ConversionResult(
            success=True,
            circuit=qasm_str,
            source_format=CircuitFormat.CIRQ,
            target_format=CircuitFormat.OPENQASM2,
        )
    except Exception as exc:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.CIRQ,
            target_format=CircuitFormat.OPENQASM2,
            error=f"QASM export failed: {exc}",
        )


def _qasm2_to_cirq(qasm_str: str, preserve_measurements: bool) -> ConversionResult:
    """Convert OpenQASM 2.0 to Cirq circuit.

    Strategy: QASM2 -> Qiskit -> Cirq (using existing converters)
    """
    if importlib.util.find_spec("cirq") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM2,
            target_format=CircuitFormat.CIRQ,
            error="Cirq is not installed",
        )

    # First convert to Qiskit
    qiskit_result = _qasm2_to_qiskit(qasm_str, preserve_measurements)
    if not qiskit_result.success:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM2,
            target_format=CircuitFormat.CIRQ,
            error=f"Intermediate QASM2->Qiskit failed: {qiskit_result.error}",
        )

    # Then convert Qiskit to Cirq
    cirq_result = _qiskit_to_cirq(qiskit_result.circuit, preserve_measurements)
    if not cirq_result.success:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM2,
            target_format=CircuitFormat.CIRQ,
            error=f"Intermediate Qiskit->Cirq failed: {cirq_result.error}",
        )

    return ConversionResult(
        success=True,
        circuit=cirq_result.circuit,
        source_format=CircuitFormat.OPENQASM2,
        target_format=CircuitFormat.CIRQ,
        warnings=qiskit_result.warnings + cirq_result.warnings,
    )


def _qasm3_to_qiskit(qasm_str: str, preserve_measurements: bool) -> ConversionResult:
    """Convert OpenQASM 3.0 to Qiskit circuit.

    Note: OpenQASM 3.0 support in Qiskit is available via qiskit.qasm3 module.
    Falls back to QASM 2.0 parser with version normalization if qasm3 unavailable.
    """
    if importlib.util.find_spec("qiskit") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM3,
            target_format=CircuitFormat.QISKIT,
            error="Qiskit is not installed",
        )

    warnings: list[str] = []

    try:
        # Try qiskit.qasm3 module first (Qiskit 0.39+)
        try:
            from qiskit import qasm3

            circuit = qasm3.loads(qasm_str)
            return ConversionResult(
                success=True,
                circuit=circuit,
                source_format=CircuitFormat.OPENQASM3,
                target_format=CircuitFormat.QISKIT,
            )
        except (ImportError, AttributeError):
            # qasm3 module not available, try fallback
            warnings.append("qiskit.qasm3 not available, attempting QASM2 fallback")

        # Fallback: Try to convert QASM3 to QASM2-compatible format
        # This is a best-effort conversion for simple circuits
        qasm2_str = _normalize_qasm3_to_qasm2(qasm_str)

        from qiskit import QuantumCircuit

        circuit = QuantumCircuit.from_qasm_str(qasm2_str)
        warnings.append("Converted via QASM3->QASM2 normalization (some features may be lost)")

        return ConversionResult(
            success=True,
            circuit=circuit,
            source_format=CircuitFormat.OPENQASM3,
            target_format=CircuitFormat.QISKIT,
            warnings=warnings,
        )
    except Exception as exc:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM3,
            target_format=CircuitFormat.QISKIT,
            error=f"QASM3 import failed: {exc}",
            warnings=warnings,
        )


def _normalize_qasm3_to_qasm2(qasm3_str: str) -> str:
    """Normalize OpenQASM 3.0 string to OpenQASM 2.0 compatible format.

    This is a best-effort conversion for simple circuits. Complex QASM3
    features (classical logic, subroutines, etc.) are not supported.
    """
    lines = []

    for line in qasm3_str.strip().split("\n"):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("//"):
            lines.append(line)
            continue

        # Replace OPENQASM 3 header
        if stripped.startswith("OPENQASM 3"):
            lines.append("OPENQASM 2.0;")
            continue

        # Handle 'include' statements
        if stripped.startswith("include"):
            # Normalize to QASM2 includes
            if "stdgates.inc" in stripped:
                lines.append('include "qelib1.inc";')
            else:
                lines.append(line)
            continue

        # Handle qubit declarations (QASM3 style)
        if stripped.startswith("qubit["):
            # Convert 'qubit[n] name;' to 'qreg name[n];'
            try:
                parts = stripped.replace("qubit[", "").split("]")
                count = parts[0]
                name = parts[1].strip().rstrip(";").strip()
                lines.append(f"qreg {name}[{count}];")
            except (IndexError, ValueError):
                lines.append(line)
            continue

        # Handle bit declarations (QASM3 style)
        if stripped.startswith("bit["):
            # Convert 'bit[n] name;' to 'creg name[n];'
            try:
                parts = stripped.replace("bit[", "").split("]")
                count = parts[0]
                name = parts[1].strip().rstrip(";").strip()
                lines.append(f"creg {name}[{count}];")
            except (IndexError, ValueError):
                lines.append(line)
            continue

        # Pass through other lines (gates, measurements, etc.)
        lines.append(line)

    return "\n".join(lines)


def _qasm3_to_cirq(qasm_str: str, preserve_measurements: bool) -> ConversionResult:
    """Convert OpenQASM 3.0 to Cirq circuit.

    Strategy: QASM3 -> Qiskit -> Cirq (using existing converters)
    """
    if importlib.util.find_spec("cirq") is None:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM3,
            target_format=CircuitFormat.CIRQ,
            error="Cirq is not installed",
        )

    # First convert to Qiskit
    qiskit_result = _qasm3_to_qiskit(qasm_str, preserve_measurements)
    if not qiskit_result.success:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM3,
            target_format=CircuitFormat.CIRQ,
            error=f"Intermediate QASM3->Qiskit failed: {qiskit_result.error}",
        )

    # Then convert Qiskit to Cirq
    cirq_result = _qiskit_to_cirq(qiskit_result.circuit, preserve_measurements)
    if not cirq_result.success:
        return ConversionResult(
            success=False,
            source_format=CircuitFormat.OPENQASM3,
            target_format=CircuitFormat.CIRQ,
            error=f"Intermediate Qiskit->Cirq failed: {cirq_result.error}",
        )

    return ConversionResult(
        success=True,
        circuit=cirq_result.circuit,
        source_format=CircuitFormat.OPENQASM3,
        target_format=CircuitFormat.CIRQ,
        warnings=qiskit_result.warnings + cirq_result.warnings,
    )


# ==============================================================================
# Validation Utilities
# ==============================================================================


def validate_for_backend(
    circuit: Any,
    backend_name: str,
    max_qubits: int,
    supported_gates: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate a circuit for a specific backend.

    Args:
        circuit: Circuit to validate
        backend_name: Name of target backend
        max_qubits: Maximum qubits supported
        supported_gates: List of supported gate names (optional)

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues: list[str] = []
    info = extract_circuit_info(circuit)

    # Check qubit count
    if info.num_qubits > max_qubits:
        issues.append(
            f"Circuit has {info.num_qubits} qubits, but {backend_name} supports max {max_qubits}"
        )

    # Check supported gates
    if supported_gates:
        supported_set = {g.lower() for g in supported_gates}
        for gate in info.gate_types:
            if gate.lower() not in supported_set:
                issues.append(f"Gate '{gate}' may not be supported by {backend_name}")

    return len(issues) == 0, issues
