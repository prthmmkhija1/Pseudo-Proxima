"""Result normalization utilities for cross-backend consistency."""

from __future__ import annotations

from typing import Any

import numpy as np

from proxima.backends.base import ExecutionResult, ResultType


def normalize_counts(
    counts: dict[str, int], *, little_endian: bool = True
) -> dict[str, float]:
    """Normalize measurement counts to probabilities.

    Args:
        counts: Raw counts dict (bitstring -> count).
        little_endian: If True, ensure output bitstrings are little-endian ordered.

    Returns:
        Dict mapping bitstring to probability (summing to 1.0).
    """
    total = sum(counts.values())
    if total == 0:
        return {}
    probabilities: dict[str, float] = {}
    for bitstring, count in counts.items():
        key = bitstring if little_endian else bitstring[::-1]
        probabilities[key] = probabilities.get(key, 0.0) + count / total
    return dict(sorted(probabilities.items()))


def normalize_statevector(statevector: Any) -> np.ndarray:
    """Ensure statevector is a normalized numpy array.

    Args:
        statevector: Raw statevector (list, ndarray, or Qiskit Statevector).

    Returns:
        Complex numpy array normalized to unit norm.
    """
    if hasattr(statevector, "data"):
        arr = np.asarray(statevector.data, dtype=complex)
    else:
        arr = np.asarray(statevector, dtype=complex)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def normalize_density_matrix(density_matrix: Any) -> np.ndarray:
    """Ensure density matrix is a valid, trace-1 numpy array.

    Args:
        density_matrix: Raw density matrix (ndarray or backend-specific object).

    Returns:
        Complex numpy array with trace normalized to 1.0.
    """
    if hasattr(density_matrix, "data"):
        arr = np.asarray(density_matrix.data, dtype=complex)
    else:
        arr = np.asarray(density_matrix, dtype=complex)
    trace = np.trace(arr)
    if trace != 0:
        arr = arr / trace
    return arr


def probabilities_from_statevector(statevector: np.ndarray) -> dict[str, float]:
    """Extract probabilities from a statevector.

    Args:
        statevector: Normalized complex statevector array.

    Returns:
        Dict mapping bitstring (little-endian) to probability.
    """
    probs = np.abs(statevector) ** 2
    n_qubits = int(np.log2(len(probs)))
    result: dict[str, float] = {}
    for idx, p in enumerate(probs):
        if p > 1e-15:
            bitstring = format(idx, f"0{n_qubits}b")[::-1]  # little-endian
            result[bitstring] = float(p)
    return dict(sorted(result.items()))


def probabilities_from_density_matrix(density_matrix: np.ndarray) -> dict[str, float]:
    """Extract probabilities from a density matrix (diagonal).

    Args:
        density_matrix: Normalized density matrix array.

    Returns:
        Dict mapping bitstring (little-endian) to probability.
    """
    diag = np.real(np.diag(density_matrix))
    n_qubits = int(np.log2(len(diag)))
    result: dict[str, float] = {}
    for idx, p in enumerate(diag):
        if p > 1e-15:
            bitstring = format(idx, f"0{n_qubits}b")[::-1]  # little-endian
            result[bitstring] = float(p)
    return dict(sorted(result.items()))


def normalize_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize an ExecutionResult for cross-backend consistency.

    - Counts are converted to probabilities.
    - Statevectors are unit-normalized.
    - Density matrices are trace-normalized.
    - Probabilities are added to metadata for quick access.

    Args:
        result: Raw ExecutionResult from a backend.

    Returns:
        ExecutionResult with normalized data and enriched metadata.
    """
    data = dict(result.data)
    metadata = dict(result.metadata)

    if result.result_type == ResultType.COUNTS and "counts" in data:
        raw_counts = data["counts"]
        probabilities = normalize_counts(raw_counts)
        data["probabilities"] = probabilities
        metadata["total_shots"] = sum(raw_counts.values())

    elif result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = normalize_statevector(data["statevector"])
        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)

    elif result.result_type == ResultType.DENSITY_MATRIX and "density_matrix" in data:
        dm = normalize_density_matrix(data["density_matrix"])
        data["density_matrix"] = dm
        data["probabilities"] = probabilities_from_density_matrix(dm)

    return ExecutionResult(
        backend=result.backend,
        simulator_type=result.simulator_type,
        execution_time_ms=result.execution_time_ms,
        qubit_count=result.qubit_count,
        shot_count=result.shot_count,
        result_type=result.result_type,
        data=data,
        metadata=metadata,
        raw_result=result.raw_result,
    )


def compare_probabilities(
    probs_a: dict[str, float],
    probs_b: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> tuple[bool, dict[str, Any]]:
    """Compare two probability distributions.

    Args:
        probs_a: First probability dict.
        probs_b: Second probability dict.
        tolerance: Max allowed difference per state.

    Returns:
        Tuple of (match: bool, details: dict with max_diff, differing_states).
    """
    all_keys = set(probs_a.keys()) | set(probs_b.keys())
    max_diff = 0.0
    differing: list[str] = []
    for key in all_keys:
        pa = probs_a.get(key, 0.0)
        pb = probs_b.get(key, 0.0)
        diff = abs(pa - pb)
        if diff > max_diff:
            max_diff = diff
        if diff > tolerance:
            differing.append(key)
    match = len(differing) == 0
    return match, {"max_diff": max_diff, "differing_states": differing}


# =============================================================================
# BACKEND-SPECIFIC NORMALIZATION FUNCTIONS
# =============================================================================


def normalize_quest_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result from the QuEST backend.
    
    QuEST-specific handling:
    - Statevectors may have different precision (single/double/quad)
    - Density matrices from mixed state simulations need trace normalization
    - GPU results may have slight numerical differences
    
    Args:
        result: ExecutionResult from QuEST backend
        
    Returns:
        Normalized ExecutionResult with consistent data format
    """
    if result.backend != "quest":
        return normalize_result(result)
    
    data = dict(result.data)
    metadata = dict(result.metadata)
    
    # Handle precision differences
    precision = metadata.get("quest_precision", "double")
    
    if result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = data["statevector"]
        
        # Convert to double precision for consistency
        if precision == "single":
            sv = np.asarray(sv, dtype=np.complex128)
        
        # Normalize and clean up small values
        sv = normalize_statevector(sv)
        
        # Zero out very small values (numerical noise)
        threshold = 1e-14 if precision == "double" else 1e-6
        sv = np.where(np.abs(sv) < threshold, 0, sv)
        
        # Re-normalize after cleanup
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm
        
        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)
    
    elif result.result_type == ResultType.DENSITY_MATRIX and "density_matrix" in data:
        dm = data["density_matrix"]
        
        # Convert to double precision
        if precision == "single":
            dm = np.asarray(dm, dtype=np.complex128)
        
        # Normalize trace to 1
        dm = normalize_density_matrix(dm)
        
        # Ensure Hermiticity (correct for numerical errors)
        dm = (dm + dm.conj().T) / 2
        
        # Zero out very small off-diagonal elements
        threshold = 1e-14 if precision == "double" else 1e-6
        dm = np.where(np.abs(dm) < threshold, 0, dm)
        
        data["density_matrix"] = dm
        data["probabilities"] = probabilities_from_density_matrix(dm)
    
    elif result.result_type == ResultType.COUNTS and "counts" in data:
        # Standard count normalization
        data["probabilities"] = normalize_counts(data["counts"])
        metadata["total_shots"] = sum(data["counts"].values())
    
    metadata["normalized_by"] = "quest_normalizer"
    
    return ExecutionResult(
        backend=result.backend,
        simulator_type=result.simulator_type,
        execution_time_ms=result.execution_time_ms,
        qubit_count=result.qubit_count,
        shot_count=result.shot_count,
        result_type=result.result_type,
        data=data,
        metadata=metadata,
        raw_result=result.raw_result,
    )


def normalize_cuquantum_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result from the cuQuantum backend.
    
    cuQuantum-specific handling:
    - GPU memory transfers may introduce precision differences
    - Results are always statevector (cuQuantum via Aer)
    - May need endianness correction for measurement results
    
    Args:
        result: ExecutionResult from cuQuantum backend
        
    Returns:
        Normalized ExecutionResult with consistent data format
    """
    if result.backend != "cuquantum":
        return normalize_result(result)
    
    data = dict(result.data)
    metadata = dict(result.metadata)
    
    # cuQuantum typically uses double precision
    precision = metadata.get("precision", "double")
    
    if result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = data["statevector"]
        
        # Ensure numpy array with correct dtype
        sv = np.asarray(sv, dtype=np.complex128)
        
        # Normalize
        sv = normalize_statevector(sv)
        
        # Clean up GPU numerical noise (slightly higher threshold)
        threshold = 1e-12
        sv = np.where(np.abs(sv) < threshold, 0, sv)
        
        # Re-normalize
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm
        
        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)
    
    elif result.result_type == ResultType.COUNTS and "counts" in data:
        counts = data["counts"]
        
        # cuQuantum via Qiskit may use big-endian, convert to little-endian
        # if needed for consistency
        is_big_endian = metadata.get("endianness", "little") == "big"
        probabilities = normalize_counts(counts, little_endian=not is_big_endian)
        
        data["probabilities"] = probabilities
        metadata["total_shots"] = sum(counts.values())
    
    metadata["normalized_by"] = "cuquantum_normalizer"
    metadata["gpu_normalized"] = True
    
    return ExecutionResult(
        backend=result.backend,
        simulator_type=result.simulator_type,
        execution_time_ms=result.execution_time_ms,
        qubit_count=result.qubit_count,
        shot_count=result.shot_count,
        result_type=result.result_type,
        data=data,
        metadata=metadata,
        raw_result=result.raw_result,
    )


def normalize_qsim_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result from the qsim backend.
    
    qsim-specific handling:
    - qsim uses Cirq conventions (big-endian qubit ordering)
    - Results may include amplitude information
    - Gate fusion may affect intermediate state representation
    
    Args:
        result: ExecutionResult from qsim backend
        
    Returns:
        Normalized ExecutionResult with consistent data format
    """
    if result.backend != "qsim":
        return normalize_result(result)
    
    data = dict(result.data)
    metadata = dict(result.metadata)
    
    if result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = data["statevector"]
        
        # Ensure numpy array
        sv = np.asarray(sv, dtype=np.complex128)
        
        # qsim uses big-endian qubit ordering (Cirq convention)
        # Convert to little-endian for consistency with other backends
        if metadata.get("qubit_order", "big_endian") == "big_endian":
            n_qubits = int(np.log2(len(sv)))
            sv = _reverse_qubit_order(sv, n_qubits)
            metadata["qubit_order_converted"] = True
        
        # Normalize
        sv = normalize_statevector(sv)
        
        # Clean up numerical noise
        threshold = 1e-14
        sv = np.where(np.abs(sv) < threshold, 0, sv)
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm
        
        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)
    
    elif result.result_type == ResultType.COUNTS and "counts" in data:
        counts = data["counts"]
        
        # qsim may return big-endian bitstrings, convert to little-endian
        converted_counts: dict[str, int] = {}
        for bitstring, count in counts.items():
            # Reverse bitstring for little-endian consistency
            le_bitstring = bitstring[::-1]
            converted_counts[le_bitstring] = converted_counts.get(le_bitstring, 0) + count
        
        data["counts"] = converted_counts
        data["probabilities"] = normalize_counts(converted_counts)
        metadata["total_shots"] = sum(converted_counts.values())
        metadata["bitstring_order_converted"] = True
    
    metadata["normalized_by"] = "qsim_normalizer"
    
    return ExecutionResult(
        backend=result.backend,
        simulator_type=result.simulator_type,
        execution_time_ms=result.execution_time_ms,
        qubit_count=result.qubit_count,
        shot_count=result.shot_count,
        result_type=result.result_type,
        data=data,
        metadata=metadata,
        raw_result=result.raw_result,
    )


def _reverse_qubit_order(statevector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reverse qubit ordering in a statevector (big-endian <-> little-endian).
    
    This function permutes the statevector amplitudes to convert between
    big-endian and little-endian qubit ordering conventions.
    
    Args:
        statevector: Input statevector
        n_qubits: Number of qubits
        
    Returns:
        Statevector with reversed qubit ordering
    """
    result = np.zeros_like(statevector)
    num_states = len(statevector)
    
    for i in range(num_states):
        # Reverse the bits
        reversed_i = int(format(i, f"0{n_qubits}b")[::-1], 2)
        result[reversed_i] = statevector[i]
    
    return result


def normalize_lret_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result from the LRET backend.
    
    LRET-specific handling:
    - Custom gate implementations may have different conventions
    - Support for native LRET result formats
    
    Args:
        result: ExecutionResult from LRET backend
        
    Returns:
        Normalized ExecutionResult with consistent data format
    """
    if result.backend != "lret":
        return normalize_result(result)
    
    data = dict(result.data)
    metadata = dict(result.metadata)
    
    if result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = normalize_statevector(data["statevector"])
        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)
    
    elif result.result_type == ResultType.COUNTS and "counts" in data:
        data["probabilities"] = normalize_counts(data["counts"])
        metadata["total_shots"] = sum(data["counts"].values())
    
    metadata["normalized_by"] = "lret_normalizer"
    
    return ExecutionResult(
        backend=result.backend,
        simulator_type=result.simulator_type,
        execution_time_ms=result.execution_time_ms,
        qubit_count=result.qubit_count,
        shot_count=result.shot_count,
        result_type=result.result_type,
        data=data,
        metadata=metadata,
        raw_result=result.raw_result,
    )


def normalize_backend_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result using backend-specific normalization.
    
    This function automatically selects the appropriate normalizer
    based on the backend name in the result.
    
    Args:
        result: ExecutionResult from any supported backend
        
    Returns:
        Normalized ExecutionResult
    """
    normalizers = {
        "quest": normalize_quest_result,
        "cuquantum": normalize_cuquantum_result,
        "qsim": normalize_qsim_result,
        "lret": normalize_lret_result,
    }
    
    normalizer = normalizers.get(result.backend, normalize_result)
    return normalizer(result)
