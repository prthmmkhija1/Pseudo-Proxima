import os
import re

base_path = r"C:\Users\dell\Pictures\intern\ProximA\Pseudo-Proxima\src\proxima\backends"
files = ["lret.py", "cirq_adapter.py", "qiskit_adapter.py", "cuquantum_adapter.py", "quest_adapter.py", "qsim_adapter.py"]

def check_feature(content, patterns):
    """Check if any pattern exists in content."""
    for pattern in patterns if isinstance(patterns, list) else [patterns]:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False

print("=" * 80)
print("PROXIMA BACKEND COMPLETION ANALYSIS")
print("=" * 80)

for filename in files:
    filepath = os.path.join(base_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.count('\n') + 1
        classes = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        
        print(f"\n{'=' * 60}")
        print(f"FILE: {filename}")
        print(f"{'=' * 60}")
        print(f"Lines: {lines}")
        print(f"Classes: {classes}")
        print()
        
        # Core methods
        print("CORE METHODS:")
        print(f"  execute(): {'YES' if check_feature(content, r'def execute\s*\(') else 'NO'}")
        print(f"  validate_circuit(): {'YES' if check_feature(content, r'def validate_circuit\s*\(') else 'NO'}")
        print(f"  estimate_resources(): {'YES' if check_feature(content, r'def estimate_resources\s*\(') else 'NO'}")
        print(f"  get_capabilities(): {'YES' if check_feature(content, r'def get_capabilities\s*\(') else 'NO'}")
        print()
        
        # Simulation features
        print("SIMULATION FEATURES:")
        print(f"  StateVector: {'YES' if check_feature(content, [r'StateVector', r'statevector', r'state_vector']) else 'NO'}")
        print(f"  DensityMatrix: {'YES' if check_feature(content, [r'DensityMatrix', r'density_matrix']) else 'NO'}")
        print(f"  Noise Model: {'YES' if check_feature(content, [r'noise', r'NoiseModel']) else 'NO'}")
        print()
        
        # Performance features
        print("PERFORMANCE FEATURES:")
        print(f"  Batch Execution: {'YES' if check_feature(content, [r'batch', r'execute_batch', r'batch_execute']) else 'NO'}")
        print(f"  GPU Support: {'YES' if check_feature(content, [r'\bgpu\b', r'\bcuda\b', r'cuQuantum', r'cuStateVec']) else 'NO'}")
        print(f"  Multi-GPU: {'YES' if check_feature(content, [r'multi.*gpu', r'device_id', r'GPUPool', r'gpu_pool']) else 'NO'}")
        print(f"  Memory Pool: {'YES' if check_feature(content, [r'memory.*pool', r'MemoryPool', r'pool_alloc']) else 'NO'}")
        print(f"  Memory Management: {'YES' if check_feature(content, [r'memory', r'Memory']) else 'NO'}")
        print()
        
        # Backend-specific features
        print("BACKEND-SPECIFIC:")
        if "cuquantum" in filename.lower():
            print(f"  cuStateVec: {'YES' if 'custatevec' in content.lower() else 'NO'}")
            print(f"  cuTensorNet: {'YES' if 'cutensornet' in content.lower() else 'NO'}")
            print(f"  CUDA Streams: {'YES' if 'stream' in content.lower() else 'NO'}")
        elif "quest" in filename.lower():
            print(f"  MPI Support: {'YES' if check_feature(content, [r'\bMPI\b', r'mpi_']) else 'NO'}")
            print(f"  OpenMP: {'YES' if check_feature(content, [r'OpenMP', r'openmp', r'num_threads', r'OMP']) else 'NO'}")
            print(f"  Precision Config: {'YES' if check_feature(content, [r'precision', r'qreal']) else 'NO'}")
        elif "qsim" in filename.lower():
            print(f"  Vectorization (AVX): {'YES' if check_feature(content, [r'AVX', r'avx', r'vector']) else 'NO'}")
            print(f"  Gate Fusion: {'YES' if check_feature(content, [r'fusion', r'fuse']) else 'NO'}")
            print(f"  Thread Control: {'YES' if check_feature(content, [r'num_threads', r'thread']) else 'NO'}")
        elif "cirq" in filename.lower():
            print(f"  cirq.Simulator: {'YES' if 'Simulator' in content else 'NO'}")
            print(f"  DensityMatrixSimulator: {'YES' if 'DensityMatrix' in content else 'NO'}")
        elif "qiskit" in filename.lower():
            print(f"  AerSimulator: {'YES' if 'AerSimulator' in content or 'Aer' in content else 'NO'}")
            print(f"  NoiseModel: {'YES' if 'NoiseModel' in content else 'NO'}")
            print(f"  GPU Backend: {'YES' if check_feature(content, [r'gpu', r'GPU']) else 'NO'}")
        elif "lret" in filename.lower():
            print(f"  Mock Simulator: {'YES' if 'MockLRET' in content else 'NO'}")
            print(f"  Gate Application: {'YES' if '_apply_' in content else 'NO'}")
        
        # Error handling
        print()
        print("ERROR HANDLING:")
        print(f"  Custom Exceptions: {'YES' if check_feature(content, [r'BackendError', r'Exception', r'raise']) else 'NO'}")
        print(f"  Exception Import: {'YES' if 'from.*exceptions import' in content or 'BackendError' in content else 'NO'}")
        
    else:
        print(f"\n{filename} NOT FOUND")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
