# GPU Computing in Julia

Comprehensive guide to GPU programming in Julia across NVIDIA, AMD, and Apple platforms.

## Overview

Julia provides multiple GPU backends:
- **CUDA.jl**: NVIDIA GPUs (most mature)
- **Metal.jl**: Apple Silicon GPUs
- **AMDGPU.jl**: AMD GPUs
- **KernelAbstractions.jl**: Portable code across all backends
- **oneAPI.jl**: Intel GPUs (experimental)

## CUDA.jl - NVIDIA GPUs

### Basic Array Operations

```julia
using CUDA

# Create GPU arrays
x = CUDA.rand(1000)           # Random array on GPU
y = CuArray(x_cpu)            # Transfer from CPU
z = CUDA.zeros(100, 100)      # Zeros on GPU

# Array operations (automatic GPU execution)
w = x .+ y                    # Element-wise add
A = CUDA.rand(100, 100)
b = A * x                     # Matrix-vector multiply (cuBLAS)

# Transfer back to CPU
x_cpu = Array(x)
```

### Custom CUDA Kernels

```julia
function gpu_kernel!(y, x, α)
    # Get thread index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Bounds check
    if i <= length(x)
        @inbounds y[i] = α * x[i]
    end

    return nothing
end

# Launch kernel
x = CUDA.rand(10000)
y = similar(x)
α = 2.0f0

threads = 256
blocks = cld(length(x), threads)  # Ceiling division

@cuda threads=threads blocks=blocks gpu_kernel!(y, x, α)

# Synchronize (wait for GPU to finish)
CUDA.@sync @cuda threads=threads blocks=blocks gpu_kernel!(y, x, α)
```

### Thread Hierarchy

```julia
# 1D grid
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

# 2D grid
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

# 3D grid
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
```

### Shared Memory

```julia
function shared_memory_kernel!(output, input)
    # Allocate shared memory
    shmem = @cuDynamicSharedMem(Float32, 256)

    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid

    # Load to shared memory
    if i <= length(input)
        @inbounds shmem[tid] = input[i]
    end

    # Synchronize threads in block
    sync_threads()

    # Use shared memory
    if i <= length(output)
        @inbounds output[i] = shmem[tid] * 2
    end

    return nothing
end

# Launch with shared memory
threads = 256
blocks = cld(length(x), threads)
shmem_size = threads * sizeof(Float32)

@cuda threads=threads blocks=blocks shmem=shmem_size shared_memory_kernel!(y, x)
```

### Reductions

```julia
using CUDA

# Built-in reductions
total = sum(x_gpu)
maximum_val = maximum(x_gpu)
result = mapreduce(f, op, x_gpu)

# Custom reduction kernel
function reduce_kernel!(output, input)
    shmem = @cuDynamicSharedMem(Float32, 256)

    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid

    # Load and reduce in shared memory
    if i <= length(input)
        @inbounds shmem[tid] = input[i]
    else
        shmem[tid] = 0.0f0
    end

    sync_threads()

    # Tree reduction in shared memory
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            @inbounds shmem[tid] += shmem[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # Write block result
    if tid == 1
        @inbounds output[blockIdx().x] = shmem[1]
    end

    return nothing
end
```

### Performance Tips

```julia
# 1. Use appropriate precision (Float32 faster than Float64 on most GPUs)
x = CuArray{Float32}(x_cpu)

# 2. Minimize CPU-GPU transfers
y_gpu = process_on_gpu(x_gpu)  # Keep on GPU

# 3. Coalesce memory access (threads access contiguous memory)
# GOOD: threads access consecutive elements
@inbounds y[i] = x[i]

# BAD: threads access strided elements
@inbounds y[i] = x[i * stride]

# 4. Occupancy - balance threads per block
threads = 256  # Often good default (32-1024 range)

# 5. Use built-in functions when available
y = x .* 2.0f0  # Uses optimized kernels
```

## Metal.jl - Apple Silicon

```julia
using Metal

# Very similar API to CUDA.jl
x = Metal.rand(1000)
y = Metal.zeros(1000)

# Custom Metal kernels
function metal_kernel!(y, x, α)
    i = thread_position_in_grid_1d()
    if i <= length(x)
        @inbounds y[i] = α * x[i]
    end
    return nothing
end

@metal threads=256 grid=length(x) metal_kernel!(y, x, 2.0f0)
```

**Differences from CUDA:**
- `thread_position_in_grid_1d()` instead of computing from block/thread idx
- `@metal` instead of `@cuda`
- Otherwise very similar API

## KernelAbstractions.jl - Portable GPU Code

Write once, run on CPU, CUDA, Metal, ROCm:

```julia
using KernelAbstractions

@kernel function my_kernel!(output, input, @Const(α))
    i = @index(Global)
    @inbounds output[i] = α * input[i]
end

# Works with any backend
function process_data(x, α)
    backend = get_backend(x)
    y = similar(x)

    kernel! = my_kernel!(backend)
    kernel!(y, x, α, ndrange=length(x))
    KernelAbstractions.synchronize(backend)

    return y
end

# Use with CUDA
x_cuda = CUDA.rand(1000)
y_cuda = process_data(x_cuda, 2.0f0)

# Use with Metal
x_metal = Metal.rand(1000)
y_metal = process_data(x_metal, 2.0f0)

# Use with CPU
x_cpu = rand(1000)
y_cpu = process_data(x_cpu, 2.0f0)
```

**Key KernelAbstractions features:**
- `@kernel`: Define portable kernel
- `@index(Global)`: Get global thread index
- `@Const`: Mark read-only arguments
- `get_backend(array)`: Get compute backend
- `ndrange`: Specify work size

## GPU-Aware Libraries

Many Julia libraries automatically support GPUs:

### DifferentialEquations.jl

```julia
using DifferentialEquations, CUDA

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

# GPU version - just use CuArray!
u0 = CuArray([1.0f0, 0.0f0, 0.0f0])
p = CuArray([10.0f0, 28.0f0, 8.0f0/3.0f0])
tspan = (0.0f0, 100.0f0)

prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Tsit5())  # Automatically runs on GPU!
```

### Flux.jl (Neural Networks)

```julia
using Flux, CUDA

# Define model
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10),
    softmax
) |> gpu  # Move to GPU

# Train on GPU
loss(x, y) = Flux.crossentropy(model(x), y)

x_train = gpu(x_train)  # Move data to GPU
y_train = gpu(y_train)

Flux.train!(loss, params(model), data, opt)
```

### Optimization.jl

```julia
using Optimization, OptimizationOptimJL, CUDA

# GPU-based objective function
function objective(x, p)
    x_gpu = CuArray(x)
    # ... GPU computations
    return Array(result)[1]  # Return scalar
end

prob = OptimizationProblem(objective, x0)
sol = solve(prob, BFGS())
```

## GPU-Specific Patterns

### Batch Processing

```julia
# Process multiple problems simultaneously
function batch_process(problems::Vector)
    # Stack into single GPU array
    batch = hcat(problems...)  # [features × batch_size]
    batch_gpu = CuArray(batch)

    # Single GPU kernel processes all
    results_gpu = process(batch_gpu)

    # Split results
    return [results_gpu[:, i] for i in 1:size(batch, 2)]
end
```

### Streaming

```julia
# Overlap computation and data transfer
stream1 = CuStream()
stream2 = CuStream()

# Process in chunks
for i in 1:nchunks
    # Transfer chunk i to GPU
    CUDA.@sync stream=stream1 begin
        x_gpu = CuArray(x_chunk)
    end

    # Process chunk i-1 while transferring
    CUDA.@sync stream=stream2 begin
        @cuda threads=threads blocks=blocks kernel!(y_gpu, x_gpu)
    end

    # Overlap!
end
```

### Pinned Memory

```julia
# Pinned memory for faster transfers
x_pinned = CUDA.Mem.pin(x_cpu)  # Pin CPU memory
x_gpu = CuArray(x_pinned)        # Faster transfer
```

## Profiling GPU Code

### CUDA.@time

```julia
CUDA.@time y = x .* 2.0f0  # Shows GPU time and memory
```

### Profile.jl with GPU

```julia
using Profile, CUDA

CUDA.@profile begin
    # GPU code here
end
```

### NVIDIA Nsight Systems

```bash
# Run Julia with profiling
nsys profile -o profile julia script.jl

# View in Nsight Systems GUI
nsys-ui profile.qdrep
```

Shows:
- Kernel launches
- Memory transfers
- CPU-GPU synchronization
- Timeline view

## Debugging GPU Code

### Scalar Indexing (Common Error)

```julia
# ERROR: scalar getindex not allowed
x = CuArray([1,2,3])
val = x[1]  # Not allowed by default (slow)

# Solution 1: Transfer to CPU
val = Array(x)[1]

# Solution 2: Allow scalar (only for debugging!)
CUDA.allowscalar() do
    val = x[1]
end

# Solution 3: Use @allowscalar macro
val = CUDA.@allowscalar x[1]
```

### Kernel Errors

```julia
# GPU kernels fail silently by default
@cuda kernel!(y, x)
CUDA.synchronize()  # Force error checking

# Or use @sync
CUDA.@sync @cuda kernel!(y, x)  # Throws error immediately
```

### Debugging Tools

```julia
# Check CUDA installation
CUDA.versioninfo()

# Check GPU memory usage
CUDA.memory_status()

# Free memory
GC.gc(true)
CUDA.reclaim()

# Device properties
dev = device()
CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
```

## GPU Best Practices Summary

**Performance:**
1. Use Float32 (faster on most GPUs)
2. Minimize CPU-GPU transfers
3. Coalesce memory access
4. Fuse operations (avoid intermediate arrays)
5. Use built-in functions (cuBLAS, etc.)
6. Profile to find bottlenecks

**Memory:**
1. Pre-allocate GPU arrays
2. Use in-place operations
3. Free memory explicitly if needed
4. Watch for memory fragmentation

**Correctness:**
1. Synchronize before checking results
2. Avoid scalar indexing
3. Check bounds in kernels
4. Test on CPU first
5. Use `CUDA.@sync` to catch errors

**Portability:**
1. Use KernelAbstractions.jl for cross-platform code
2. Write generic algorithms that work with any array type
3. Test on multiple backends
4. Avoid backend-specific features unless necessary

## Common Pitfalls

1. **Forgetting to synchronize**: GPU kernels are asynchronous
2. **Scalar indexing**: Very slow, use `Array()` to transfer
3. **Type instability on GPU**: Even more expensive than CPU
4. **Too many/few threads**: Tune threads per block
5. **Excessive memory transfers**: Keep data on GPU
6. **Not using Float32**: Float64 is often 2-32× slower
7. **Ignoring errors**: Use `CUDA.@sync` to catch errors

## When to Use GPU

**Good for:**
- Large data parallel workloads
- Dense linear algebra (matrix multiply, etc.)
- Element-wise operations on large arrays
- Deep learning / neural networks
- Monte Carlo simulations
- Image/signal processing

**Bad for:**
- Small data (< 10k elements)
- Highly branching code
- Sequential algorithms
- Frequent CPU-GPU transfers
- Sparse, irregular computations
