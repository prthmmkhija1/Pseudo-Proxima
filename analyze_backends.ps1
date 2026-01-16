$basePath = "C:\Users\dell\Pictures\intern\ProximA\Pseudo-Proxima\src\proxima\backends"
$files = @("lret.py", "cirq_adapter.py", "qiskit_adapter.py", "cuquantum_adapter.py", "quest_adapter.py", "qsim_adapter.py")

foreach ($f in $files) {
    $path = Join-Path $basePath $f
    if (Test-Path $path) {
        $content = Get-Content $path -Raw
        $lineCount = (Get-Content $path).Count
        
        Write-Host ""
        Write-Host "========== $f =========="
        Write-Host "Lines: $lineCount"
        
        # Check for key features
        $hasExecute = $content -match 'def execute\s*\('
        $hasValidate = $content -match 'def validate_circuit\s*\(|validate_circuit'
        $hasEstimate = $content -match 'def estimate_resources\s*\(|estimate_resources'
        $hasCaps = $content -match 'def get_capabilities\s*\(|get_capabilities'
        $hasDensity = $content -match 'DensityMatrix|density_matrix'
        $hasStateVec = $content -match 'StateVector|statevector'
        $hasGPU = $content -match 'gpu|cuda|GPU|CUDA|cuQuantum'
        $hasNoise = $content -match 'noise|NoiseModel'
        $hasBatch = $content -match 'batch|execute_batch|batch_execute'
        $hasMemory = $content -match 'memory|Memory'
        $hasMPI = $content -match 'MPI|mpi'
        $hasOpenMP = $content -match 'OpenMP|openmp|OMP|num_threads'
        $hasMultiGPU = $content -match 'multi.*gpu|GPU.*pool|device.*id|MultiGPU'
        $hasMemPool = $content -match 'memory.*pool|MemoryPool|pool.*alloc'
        $hasPrecision = $content -match 'precision|double|float32|float64'
        
        Write-Host "execute method: $(if($hasExecute){'YES'}else{'NO'})"
        Write-Host "validate_circuit: $(if($hasValidate){'YES'}else{'NO'})"
        Write-Host "estimate_resources: $(if($hasEstimate){'YES'}else{'NO'})"
        Write-Host "get_capabilities: $(if($hasCaps){'YES'}else{'NO'})"
        Write-Host "DensityMatrix: $(if($hasDensity){'YES'}else{'NO'})"
        Write-Host "StateVector: $(if($hasStateVec){'YES'}else{'NO'})"
        Write-Host "GPU support: $(if($hasGPU){'YES'}else{'NO'})"
        Write-Host "Noise model: $(if($hasNoise){'YES'}else{'NO'})"
        Write-Host "Batch execution: $(if($hasBatch){'YES'}else{'NO'})"
        Write-Host "Memory mgmt: $(if($hasMemory){'YES'}else{'NO'})"
        Write-Host "MPI support: $(if($hasMPI){'YES'}else{'NO'})"
        Write-Host "OpenMP/Threading: $(if($hasOpenMP){'YES'}else{'NO'})"
        Write-Host "Multi-GPU: $(if($hasMultiGPU){'YES'}else{'NO'})"
        Write-Host "Memory Pool: $(if($hasMemPool){'YES'}else{'NO'})"
        Write-Host "Precision config: $(if($hasPrecision){'YES'}else{'NO'})"
    } else {
        Write-Host "$f NOT FOUND"
    }
}
