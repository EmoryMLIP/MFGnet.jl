#!/usr/bin/env julia
# Test and verify GPU setup for Julia development

using Pkg

println("🔍 Checking GPU setup for Julia...")
println("="^60)

# Check for CUDA
println("\n📦 Checking CUDA.jl...")
try
    Pkg.add("CUDA")
    using CUDA

    if CUDA.functional()
        println("✅ CUDA.jl is functional!")
        println("   CUDA version: ", CUDA.version())
        println("   Driver version: ", CUDA.driver_version())
        println("   Runtime version: ", CUDA.runtime_version())

        # Get device info
        if CUDA.has_cuda_gpu()
            device = CUDA.device()
            println("   Device: ", CUDA.name(device))
            println("   Compute capability: ", CUDA.capability(device))
            total_mem = CUDA.totalmem(device) / 1024^3
            println("   Total memory: ", round(total_mem, digits=2), " GB")

            # Run a simple test
            println("\n🧪 Running simple GPU test...")
            x = CUDA.rand(1000, 1000)
            y = x * x
            CUDA.@sync y  # Synchronize
            println("✅ GPU computation successful!")
        else
            println("⚠️  CUDA installed but no GPU detected")
        end
    else
        println("❌ CUDA.jl installed but not functional")
        println("   This might be due to:")
        println("   - No NVIDIA GPU present")
        println("   - Missing or incompatible CUDA drivers")
        println("   - Run CUDA.versioninfo() for details")
    end
catch e
    println("❌ CUDA.jl not available: ", e)
    println("   To install: julia -e 'using Pkg; Pkg.add(\"CUDA\")'")
end

# Check for AMDGPU
println("\n📦 Checking AMDGPU.jl...")
try
    Pkg.add("AMDGPU")
    using AMDGPU

    if AMDGPU.functional()
        println("✅ AMDGPU.jl is functional!")
        println("   ROCm version: ", AMDGPU.runtime_version())
    else
        println("⚠️  AMDGPU.jl installed but not functional")
    end
catch e
    println("ℹ️  AMDGPU.jl not available (expected if no AMD GPU)")
end

# Check for Metal (Apple Silicon)
println("\n📦 Checking Metal.jl...")
try
    Pkg.add("Metal")
    using Metal

    if Metal.functional()
        println("✅ Metal.jl is functional!")
        println("   Running on Apple Silicon with GPU support")

        # Run a simple test
        println("\n🧪 Running simple Metal test...")
        x = Metal.rand(1000, 1000)
        y = x * x
        Metal.@sync y
        println("✅ Metal GPU computation successful!")
    else
        println("⚠️  Metal.jl installed but not functional")
    end
catch e
    println("ℹ️  Metal.jl not available (expected if not on Apple Silicon)")
end

# Check for KernelAbstractions
println("\n📦 Checking KernelAbstractions.jl...")
try
    Pkg.add("KernelAbstractions")
    using KernelAbstractions
    println("✅ KernelAbstractions.jl available")
    println("   This enables portable GPU kernels across CUDA/ROCm/Metal")
catch e
    println("❌ KernelAbstractions.jl not available")
end

# Summary
println("\n" * "="^60)
println("📋 GPU Setup Summary:")
println("="^60)

has_gpu = false
if @isdefined(CUDA) && CUDA.functional()
    println("✅ NVIDIA GPU support: Available")
    has_gpu = true
elseif @isdefined(AMDGPU) && AMDGPU.functional()
    println("✅ AMD GPU support: Available")
    has_gpu = true
elseif @isdefined(Metal) && Metal.functional()
    println("✅ Apple GPU support: Available")
    has_gpu = true
end

if !has_gpu
    println("ℹ️  No GPU support detected")
    println("   This is fine for CPU-only development")
    println("   GPU packages will work but fall back to CPU")
end

println("\n💡 Recommended GPU packages:")
println("   - CUDA.jl: NVIDIA GPU support")
println("   - Metal.jl: Apple Silicon GPU support")
println("   - AMDGPU.jl: AMD GPU support")
println("   - KernelAbstractions.jl: Portable GPU kernels")
println("   - Tullio.jl: Fast array operations (GPU-aware)")
println("   - Flux.jl: Neural networks with GPU support")
