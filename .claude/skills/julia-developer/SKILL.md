---
name: julia-developer
description: Use this skill when developing, debugging, optimizing, or analyzing Julia code. This includes writing new Julia packages, maintaining existing codebases, debugging issues, optimizing performance, working with GPUs, implementing algorithms using automatic differentiation or differential equations, and analyzing Julia projects. Supports both VS Code and Jupyter workflows, with comprehensive knowledge of Julia best practices including JIT optimization, multiple dispatch, type stability, GPU computing, and the scientific computing ecosystem (AD, ODEs, optimization).
---

# Julia Developer

## Overview

This skill provides comprehensive support for Julia development across the entire software lifecycle: from environment setup and package creation to performance optimization and GPU computing. Use this skill for writing, debugging, testing, and optimizing Julia code with best practices for the Julia ecosystem including automatic differentiation, differential equations, optimization, and GPU computing.

## When to Use This Skill

Invoke this skill when:
- Creating new Julia packages or modules
- Debugging Julia code with breakpoints and inspection
- Optimizing performance (JIT, type stability, allocations)
- Working with GPU code (CUDA.jl, Metal.jl, KernelAbstractions.jl)
- Implementing scientific computing workflows (AD, ODEs, optimization)
- Setting up Julia development environments or dev containers
- Analyzing or maintaining existing Julia codebases
- Writing tests following Julia community standards
- Implementing algorithms using multiple dispatch patterns

## Environment Setup

### Quick Start

For local development setup, run the setup script:

```bash
bash scripts/setup_dev_environment.sh
```

This installs essential packages (Revise, OhMyREPL, BenchmarkTools, Debugger, Infiltrator) and configures the Julia startup file for an optimal development experience.

### Dev Container

For VS Code dev container setup, copy the configuration:

```bash
mkdir -p .devcontainer
cp assets/devcontainer/devcontainer.json .devcontainer/
```

The dev container provides:
- Julia 1.10+ pre-installed
- VS Code Julia extension configured
- Essential development packages pre-installed
- Consistent environment across team members

### GPU Setup

To verify and setup GPU computing:

```bash
julia scripts/gpu_setup_test.jl
```

This checks for CUDA (NVIDIA), Metal (Apple Silicon), or AMDGPU (AMD) support and provides setup guidance.

## Core Development Workflows

### 1. Creating New Packages

Use the package creation script to initialize a properly structured package:

```bash
julia scripts/create_package.jl MyPackage --user=myusername
```

For GPU-enabled packages:

```bash
julia scripts/create_package.jl MyPackage --user=myusername --gpu
```

This creates a complete package structure with:
- Standard directory layout (src/, test/, docs/)
- Proper Project.toml with dependencies
- CI/CD configuration (GitHub Actions)
- Documentation setup (Documenter.jl)
- Testing infrastructure

**Reference:** See `references/package_structure.md` for detailed package organization guidelines.

### 2. Writing Julia Code

When writing Julia code, follow these principles:

#### Type Stability (Critical for Performance)

Always ensure type stability. Check with `@code_warntype`:

```julia
@code_warntype my_function(x)
```

Look for:
- ✅ Blue concrete types (good)
- ❌ Red Union types or yellow Any (type instability)

**Common fixes:**
- Use parametric types: `struct Container{T}` not `struct Container`
- Avoid global variables or use `const`
- Ensure consistent return types
- Type struct fields: `x::Float64` not `x::AbstractFloat`

**Reference:** See `references/performance_optimization.md` for comprehensive optimization guide.

#### Multiple Dispatch

Leverage Julia's multiple dispatch for clean, performant code:

```julia
# Generic fallback
process(x::AbstractArray) = sum(x)

# Specialized for specific types
process(x::Vector{Float64}) = optimized_sum(x)
process(x::CuArray) = gpu_sum(x)
```

Write generic code first, specialize only when needed.

**Reference:** See `references/multiple_dispatch.md` for advanced patterns (traits, type hierarchies, dispatch strategies).

#### Memory Management

Minimize allocations:
- Pre-allocate arrays outside loops
- Use in-place operations (`mul!`, `copy!`, `fill!`)
- Use `@views` to avoid array copies
- Consider `StaticArrays` for small fixed-size arrays

Check allocations with:
```julia
@time my_function(x)        # Shows allocations
@allocated my_function(x)   # Returns allocation count
```

### 3. Debugging

For interactive debugging with breakpoints:

```julia
using Infiltrator

function my_function(x)
    y = complex_computation(x)
    @infiltrate  # Breakpoint: inspect variables, step through
    return process(y)
end
```

For full debugging with step-through:

```julia
using Debugger
@enter my_function(x)
```

**VS Code users:** Set breakpoints in the editor and press F5 to debug visually.

**Reference:** See `references/debugging_guide.md` for comprehensive debugging techniques including GPU debugging, profiling, and common issue patterns.

### 4. Performance Optimization

Follow this workflow for optimization:

```julia
include("scripts/profile_workflow.jl")

# 1. Profile to find bottlenecks
profile_function(my_function, x)

# 2. Benchmark current performance
benchmark_function(my_function, x)

# 3. Check allocations
check_allocations(my_function, x)

# 4. Check type stability
@code_warntype my_function(x)

# 5. Optimize and repeat
```

**Key optimization checklist:**
- [ ] Type stability verified
- [ ] Allocations minimized
- [ ] In-place operations used where possible
- [ ] Appropriate use of `@inbounds`, `@simd`
- [ ] GPU offloading considered (if applicable)

**Reference:** See `references/performance_optimization.md` for detailed optimization patterns.

### 5. Testing

Write comprehensive tests following Julia community standards:

```julia
using Test

@testset "MyPackage" begin
    @testset "Basic functionality" begin
        x = [1, 2, 3]
        @test my_function(x) == expected_result
    end

    @testset "Edge cases" begin
        @test_throws ArgumentError my_function([])
        @test my_function([0]) ≈ 0.0 atol=1e-10
    end

    @testset "Type stability" begin
        @test @inferred my_function([1.0, 2.0])
    end

    @testset "Generic code" begin
        for T in (Float32, Float64, Int64)
            x = T[1, 2, 3]
            @test my_function(x) isa T
        end
    end
end
```

Run tests:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Reference:** See `references/testing_practices.md` for comprehensive testing patterns, CI/CD setup, and coverage.

## GPU Computing

### Basic GPU Usage

For CUDA (NVIDIA) or Metal (Apple Silicon):

```julia
using CUDA  # or: using Metal

# Move data to GPU
x_gpu = CuArray(x)  # or: Metal.MtlArray(x)

# Operations automatically use GPU
y_gpu = x_gpu .* 2
result_gpu = x_gpu * A_gpu

# Transfer back to CPU
result = Array(result_gpu)
```

### Custom GPU Kernels

For custom operations:

```julia
using CUDA

function my_kernel!(output, input, α)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(input)
        @inbounds output[i] = α * input[i]
    end
    return nothing
end

threads = 256
blocks = cld(length(x), threads)
@cuda threads=threads blocks=blocks my_kernel!(y, x, 2.0f0)
CUDA.synchronize()
```

### Portable GPU Code (KernelAbstractions.jl)

Write once, run on CPU/CUDA/Metal/ROCm:

```julia
using KernelAbstractions

@kernel function my_kernel!(output, input, @Const(α))
    i = @index(Global)
    @inbounds output[i] = α * input[i]
end

backend = get_backend(x)
kernel! = my_kernel!(backend)
kernel!(output, input, α, ndrange=length(input))
```

**Reference:** See `references/gpu_computing.md` for comprehensive GPU programming guide.

## Scientific Computing Workflows

### Automatic Differentiation

Choose the appropriate AD system:

**Forward-mode (ForwardDiff.jl):** Best for f: R^n → R^m where n << m
```julia
using ForwardDiff
gradient = ForwardDiff.gradient(f, x)
```

**Reverse-mode (Zygote.jl):** Best for optimization, ML (f: R^n → R)
```julia
using Zygote
gradient = Zygote.gradient(f, x)
```

**LLVM-level (Enzyme.jl):** Fastest, handles mutation, GPU kernels
```julia
using Enzyme
Enzyme.autodiff(Reverse, f, Active, Active(x))
```

### Differential Equations

```julia
using DifferentialEquations

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)  # Automatic algorithm selection
```

**GPU support:** Just use `CuArray` for arrays and parameters.

### Optimization

```julia
using Optim

result = optimize(f, gradient!, x0, BFGS())

# Or use JuMP for constrained optimization
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)
@variable(model, x >= 0)
@objective(model, Min, f(x))
@constraint(model, g(x) <= 0)
optimize!(model)
```

**Reference:** See `references/julia_ecosystem.md` for comprehensive package guide.

## Common Patterns and Best Practices

### Code Organization

Structure packages following community standards:
- `src/PackageName.jl` - Main module file with exports
- `src/submodule.jl` - Additional functionality
- `test/runtests.jl` - Main test file including sub-tests
- `docs/` - Documenter.jl documentation
- `examples/` - Runnable example scripts

### Documentation

Write comprehensive docstrings:

```julia
\"\"\"
    my_function(x::AbstractArray; tol=1e-6)

Brief description.

# Arguments
- `x::AbstractArray`: Description
- `tol::Float64`: Tolerance (default: 1e-6)

# Returns
- `result::Float64`: Description

# Examples
```jldoctest
julia> my_function([1, 2, 3])
6.0
```

# See Also
- [`related_function`](@ref)
\"\"\"
function my_function(x; tol=1e-6)
    # Implementation
end
```

### Type Design

```julia
# Immutable when possible (stack allocated, thread-safe)
struct Point{T<:Real}
    x::T
    y::T
end

# Use parametric types for performance
struct Container{T<:AbstractArray}
    data::T  # Concrete at instantiation
end
```

### Error Handling

```julia
function validate_input(x)
    length(x) > 0 || throw(ArgumentError("x must not be empty"))
    all(isfinite, x) || throw(DomainError(x, "x must be finite"))
    return true
end
```

## Development Tips

### Use Revise.jl for Interactive Development

```julia
using Revise
using MyPackage

# Edit code in src/
# Changes automatically reload - no need to restart Julia
```

### Profiling Workflow

```julia
using Profile, ProfileView

@profile my_function(x)
Profile.print()  # Text output
@profview my_function(x)  # Visual flame graph
```

### Benchmarking

```julia
using BenchmarkTools

# Quick timing
@btime my_function($x)

# Detailed statistics
@benchmark my_function($x)
```

### Startup Configuration

Edit `~/.julia/config/startup.jl` for persistent REPL configuration:

```julia
try
    using Revise
catch e
    @warn "Revise not available"
end

try
    using OhMyREPL
catch e
    @warn "OhMyREPL not available"
end
```

## Troubleshooting Common Issues

### MethodError
Run `methods(foo)` to see available methods and `@which foo(args)` to see which would be called.

### Type Instability
Run `@code_warntype function(args)` and look for red or yellow types.

### Performance Issues
Profile first (`@profview`), then optimize bottlenecks. Focus on type stability and allocations.

### GPU Errors
Use `CUDA.@sync @cuda kernel!(...)` to catch errors immediately. Check with `CUDA.memory_status()` for memory issues.

### Package Conflicts
Run `] resolve` to resolve dependency issues. Check versions with `] status`.

## Resources

### Scripts (scripts/)
Executable tools for common tasks:
- `setup_dev_environment.sh` - Complete environment setup
- `create_package.jl` - Package initialization with templates
- `gpu_setup_test.jl` - GPU environment verification
- `profile_workflow.jl` - Profiling and benchmarking utilities

### References (references/)
Comprehensive guides (load as needed with Read tool):
- `julia_ecosystem.md` - Package ecosystem guide (AD, GPU, ODE, optimization)
- `performance_optimization.md` - JIT optimization, type stability, memory management
- `multiple_dispatch.md` - Dispatch patterns and best practices
- `gpu_computing.md` - CUDA.jl, Metal.jl, KernelAbstractions.jl
- `testing_practices.md` - Testing, coverage, CI/CD standards
- `package_structure.md` - Standard package organization
- `debugging_guide.md` - Debugging tools and techniques

### Assets (assets/)
Templates and configurations:
- `devcontainer/` - VS Code dev container configuration
- `package_template/` - Complete package structure template
- `notebook_templates/` - Jupyter/script templates for exploration

## Quick Reference

**Type stability check:**
```julia
@code_warntype my_function(x)
```

**Performance profiling:**
```julia
@btime my_function($x)
@profview my_function(x)
```

**Debugging:**
```julia
@infiltrate  # Lightweight breakpoint
@enter my_function(x)  # Full debugger
```

**Testing:**
```julia
] test MyPackage
```

**GPU check:**
```julia
CUDA.functional()  # or Metal.functional()
```

**Package development:**
```julia
] dev path/to/Package
using Revise, MyPackage
```

## Summary

This skill provides end-to-end support for Julia development with emphasis on:
1. **Performance**: Type stability, memory optimization, JIT compilation
2. **Best practices**: Multiple dispatch, generic programming, community standards
3. **Ecosystem**: AD, ODEs, optimization, GPU computing
4. **Workflow**: Development, debugging, testing, profiling, optimization
5. **Environments**: Local setup, dev containers, GPU configuration

Always start by understanding the problem, write clear generic code first, then optimize based on profiling data. Leverage Julia's strengths: multiple dispatch, JIT compilation, composability, and a rich scientific computing ecosystem.
