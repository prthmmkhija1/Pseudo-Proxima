#!/usr/bin/env python3
"""Add missing Cirq and Qiskit normalizers."""

with open('src/proxima/backends/normalization.py', 'r') as f:
    content = f.read()

# New normalizer functions
new_functions = '''

def normalize_cirq_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result from the Cirq backend.

    Cirq-specific handling:
    - Measurement results use big-endian qubit ordering by default
    - Statevector amplitudes may need reordering for consistency
    - Handles cirq.Result objects with different measurement formats

    Args:
        result: ExecutionResult from Cirq backend

    Returns:
        Normalized ExecutionResult with consistent data format
    """
    if result.backend != "cirq":
        return normalize_result(result)

    data = dict(result.data)
    metadata = dict(result.metadata)

    if result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = data["statevector"]

        # Ensure numpy array with correct dtype
        sv = np.asarray(sv, dtype=np.complex128)

        # Normalize
        sv = normalize_statevector(sv)

        # Clean up numerical noise
        threshold = 1e-14
        sv = np.where(np.abs(sv) < threshold, 0, sv)

        # Cirq uses big-endian by default, convert to little-endian for consistency
        if metadata.get("qubit_order", "big_endian") == "big_endian":
            n_qubits = result.qubit_count
            if n_qubits > 0:
                sv = _reverse_qubit_order(sv, n_qubits)
                metadata["qubit_order_converted"] = True

        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)

    elif result.result_type == ResultType.DENSITY_MATRIX and "density_matrix" in data:
        dm = data["density_matrix"]
        dm = np.asarray(dm, dtype=np.complex128)
        dm = normalize_density_matrix(dm)
        data["density_matrix"] = dm
        data["probabilities"] = probabilities_from_density_matrix(dm)

    elif result.result_type == ResultType.COUNTS and "counts" in data:
        counts = data["counts"]

        # Cirq uses big-endian qubit ordering, convert to little-endian
        converted_counts: dict[str, int] = {}
        for bitstring, count in counts.items():
            # Reverse bitstring for little-endian
            le_bitstring = bitstring[::-1]
            converted_counts[le_bitstring] = converted_counts.get(le_bitstring, 0) + count

        data["counts"] = converted_counts
        data["probabilities"] = normalize_counts(converted_counts)
        metadata["total_shots"] = sum(converted_counts.values())
        metadata["bitstring_order_converted"] = True

    metadata["normalized_by"] = "cirq_normalizer"

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


def normalize_qiskit_result(result: ExecutionResult) -> ExecutionResult:
    """Normalize a result from the Qiskit Aer backend.

    Qiskit-specific handling:
    - Qiskit uses little-endian qubit ordering (already consistent)
    - Handles Qiskit Result objects with counts and statevector
    - Supports both sampler and estimator primitives results

    Args:
        result: ExecutionResult from Qiskit Aer backend

    Returns:
        Normalized ExecutionResult with consistent data format
    """
    if result.backend not in ("qiskit", "qiskit_aer", "aer"):
        return normalize_result(result)

    data = dict(result.data)
    metadata = dict(result.metadata)

    if result.result_type == ResultType.STATEVECTOR and "statevector" in data:
        sv = data["statevector"]

        # Handle Qiskit Statevector object
        if hasattr(sv, "data"):
            sv = sv.data

        # Ensure numpy array with correct dtype
        sv = np.asarray(sv, dtype=np.complex128)

        # Normalize
        sv = normalize_statevector(sv)

        # Clean up numerical noise
        threshold = 1e-14
        sv = np.where(np.abs(sv) < threshold, 0, sv)

        data["statevector"] = sv
        data["probabilities"] = probabilities_from_statevector(sv)

    elif result.result_type == ResultType.DENSITY_MATRIX and "density_matrix" in data:
        dm = data["density_matrix"]

        # Handle Qiskit DensityMatrix object
        if hasattr(dm, "data"):
            dm = dm.data

        dm = np.asarray(dm, dtype=np.complex128)
        dm = normalize_density_matrix(dm)
        data["density_matrix"] = dm
        data["probabilities"] = probabilities_from_density_matrix(dm)

    elif result.result_type == ResultType.COUNTS and "counts" in data:
        counts = data["counts"]

        # Qiskit already uses little-endian, just normalize probabilities
        data["probabilities"] = normalize_counts(counts, little_endian=True)
        metadata["total_shots"] = sum(counts.values())

    metadata["normalized_by"] = "qiskit_normalizer"

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

'''

# Find the position to insert (before normalize_backend_result)
insert_marker = 'def normalize_backend_result(result: ExecutionResult) -> ExecutionResult:'
if insert_marker in content:
    content = content.replace(insert_marker, new_functions + insert_marker)
    
    # Also update the normalizers dictionary
    old_dict = '''    normalizers = {
        "quest": normalize_quest_result,
        "cuquantum": normalize_cuquantum_result,
        "qsim": normalize_qsim_result,
        "lret": normalize_lret_result,
    }'''
    
    new_dict = '''    normalizers = {
        "quest": normalize_quest_result,
        "cuquantum": normalize_cuquantum_result,
        "qsim": normalize_qsim_result,
        "lret": normalize_lret_result,
        "cirq": normalize_cirq_result,
        "qiskit": normalize_qiskit_result,
        "qiskit_aer": normalize_qiskit_result,
        "aer": normalize_qiskit_result,
    }'''
    
    content = content.replace(old_dict, new_dict)
    
    with open('src/proxima/backends/normalization.py', 'w') as f:
        f.write(content)
    print('Added normalize_cirq_result and normalize_qiskit_result')
    print('Updated normalizers dictionary')
else:
    print('Could not find insertion point')
