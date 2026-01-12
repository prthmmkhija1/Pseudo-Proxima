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
