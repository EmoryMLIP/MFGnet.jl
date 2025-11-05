# Julia Ecosystem Guide

Comprehensive guide to essential Julia packages for scientific computing, organized by domain.

## Automatic Differentiation (AD)

### Forward-Mode AD

**ForwardDiff.jl** - Mature, reliable forward-mode AD
```julia
using ForwardDiff

f(x) = sum(x.^2)
gradient = ForwardDiff.gradient(f, x)
hessian = ForwardDiff.hessian(f, x)
```
- Best for: Functions with few inputs, many outputs (e.g., Jacobians of f: R^n → R^m where n << m)
- Uses dual numbers for exact derivatives
- Works well with generic Julia code

**FiniteDiff.jl** - Finite difference approximations
```julia
using FiniteDiff

FiniteDiff.finite_difference_gradient(f, x)
```
- Fallback when AD fails
- Less accurate than AD but very robust

### Reverse-Mode AD

**Zygote.jl** - Source-to-source reverse-mode AD
```julia
using Zygote

gradient_f = gradient(f, x)
pullback = Zygote.pullback(f, x)
```
- Best for: Machine learning, optimization (functions R^n → R where n >> 1)
- De facto standard for Flux.jl
- Some limitations with mutation and control flow

**ChainRules.jl** - Define custom AD rules
```julia
using ChainRules

# Define custom derivative rule
function ChainRules.rrule(::typeof(myfunction), x)
    y = myfunction(x)
    function myfunction_pullback(ȳ)
        x̄ = # ... custom gradient computation
        return NoTangent(), x̄
    end
    return y, myfunction_pullback
end
```
- Write custom rules for performance or when AD fails
- Integrates with Zygote, ForwardDiff, Enzyme

**Enzyme.jl** - LLVM-level AD (cutting edge)
```julia
using Enzyme

Enzyme.autodiff(Reverse, f, Active, Active(x))
```
- Fastest AD system available
- Handles mutation and complex control flow
- Works with GPU kernels
- More complex to use than Zygote

### Choosing AD System

| Task | Recommended | Reason |
|------|------------|--------|
| ML training | Zygote.jl | Ecosystem integration |
| Optimization | ForwardDiff.jl or Zygote.jl | Depends on n vs m |
| GPU computing | Enzyme.jl | LLVM-level, GPU kernels |
| Complex code | Enzyme.jl | Handles mutation |
| Quick prototyping | ForwardDiff.jl | Easy to use |

## GPU Computing

### CUDA.jl - NVIDIA GPUs

```julia
using CUDA

# Array operations
x = CUDA.rand(1000, 1000)
y = x * x  # Automatically uses cuBLAS

# Custom kernels
function kernel!(y, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds y[i] = sqrt(x[i])
    end
    return nothing
end

threads = 256
blocks = cld(length(x), threads)
@cuda threads=threads blocks=blocks kernel!(y, x)
```

**Key functions:**
- `CuArray`: GPU array type
- `@cuda`: Launch kernels
- `CUDA.@sync`: Synchronize GPU operations
- `CUDA.@allowscalar`: Allow scalar indexing (slow, debug only)

### Metal.jl - Apple Silicon GPUs

```julia
using Metal

x = Metal.rand(1000, 1000)
y = x * x  # Uses Metal Performance Shaders
```

Similar API to CUDA.jl but for Apple GPUs.

### KernelAbstractions.jl - Portable GPU Code

```julia
using KernelAbstractions

@kernel function my_kernel!(y, x)
    i = @index(Global)
    @inbounds y[i] = sqrt(x[i])
end

# Works on CPU, CUDA, Metal, etc.
backend = get_backend(x)
kernel! = my_kernel!(backend)
kernel!(y, x, ndrange=length(x))
```

Write once, run on any backend (CPU/CUDA/ROCm/Metal).

### GPU Best Practices

1. **Minimize CPU-GPU transfers** - Keep data on GPU
2. **Use views** - Avoid copying: `@view x[1:100]`
3. **Batch operations** - One large kernel > many small kernels
4. **Match precision** - Use Float32 on GPU (usually faster)
5. **Profile** - Use CUDA.@profile or NSight Systems

## Differential Equations

### DifferentialEquations.jl - Comprehensive ODE/PDE Solver

```julia
using DifferentialEquations

# Define ODE: dy/dt = f(y, p, t)
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
p = [10.0, 28.0, 8/3]

prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)
```

**Key features:**
- Automatic stiffness detection
- GPU support (CUDA arrays)
- Automatic differentiation (sensitivity analysis)
- 300+ solver algorithms

**Common solvers:**
- `Tsit5()`: Non-stiff (default, good first choice)
- `Rodas4()`: Stiff
- `TRBDF2()`: Stiff, large systems
- `QNDF()`: Stiff, high accuracy

**Advanced features:**
```julia
# Sensitivity analysis (gradients w.r.t. parameters)
using DiffEqSensitivity

function loss(p)
    sol = solve(prob, p=p, saveat=0.1)
    sum(abs2, sol .- target)
end

gradient = ForwardDiff.gradient(loss, p)

# GPU acceleration
using CUDA
prob_gpu = ODEProblem(lorenz!, CuArray(u0), tspan, CuArray(p))
sol_gpu = solve(prob_gpu)
```

### ModelingToolkit.jl - Symbolic Modeling

```julia
using ModelingToolkit

@variables t x(t) y(t)
@parameters σ ρ β
D = Differential(t)

eqs = [
    D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z
]

@named sys = ODESystem(eqs, t)
sys = structural_simplify(sys)  # Automatic optimization
```

Symbolic manipulation, code generation, and optimization of equation systems.

## Optimization

### Optim.jl - General-Purpose Optimization

```julia
using Optim

# Minimize f(x)
result = optimize(f, gradient!, x0, BFGS())

# Common algorithms:
# - NelderMead(): Derivative-free
# - BFGS(): Quasi-Newton (unconstrained)
# - LBFGS(): Limited-memory BFGS (large-scale)
# - Newton(): Second-order (requires Hessian)
# - GradientDescent(): Simple first-order
```

### JuMP.jl - Mathematical Optimization

```julia
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)

@variable(model, x >= 0)
@variable(model, y >= 0)

@objective(model, Min, x^2 + y^2)
@constraint(model, x + y >= 1)

optimize!(model)
value(x), value(y)
```

Best for: Linear programming, convex optimization, mixed-integer programming.

### Optimization.jl - Unified Interface

```julia
using Optimization, OptimizationOptimJL

prob = OptimizationProblem(f, x0, p)
sol = solve(prob, BFGS())
```

Unified interface to multiple optimization packages.

### Choosing Optimization Tools

| Problem Type | Recommended | Package |
|-------------|-------------|---------|
| Smooth unconstrained | BFGS, L-BFGS | Optim.jl |
| Non-smooth | Nelder-Mead | Optim.jl |
| Linear programming | Simplex | JuMP.jl + Clp |
| Convex | Interior point | JuMP.jl + Ipopt |
| Mixed-integer | Branch & bound | JuMP.jl + GLPK |
| Black-box global | Genetic algorithms | BlackBoxOptim.jl |

## Linear Algebra

Julia has excellent built-in linear algebra, but these packages extend capabilities:

### LinearAlgebra.jl (stdlib)
- BLAS/LAPACK wrappers
- Matrix factorizations: `lu`, `qr`, `svd`, `eigen`, `cholesky`
- Structured matrices: `Diagonal`, `Tridiagonal`, `SymTridiagonal`

### SuiteSparse.jl (stdlib)
- Sparse matrix support
- `sparse()`, `spzeros()`, `sprand()`
- Fast sparse solvers

### IterativeSolvers.jl
```julia
using IterativeSolvers

# Solve Ax = b iteratively
x = cg(A, b)  # Conjugate gradient
x = gmres(A, b)  # GMRES
```

Best for: Large sparse systems where direct methods are too slow.

### LinearSolve.jl
```julia
using LinearSolve

prob = LinearProblem(A, b)
sol = solve(prob)  # Automatically chooses best algorithm
```

Unified interface that automatically selects the best algorithm.

## Array and Tensor Operations

### Tullio.jl - Einstein Notation

```julia
using Tullio

# Matrix multiply: C[i,k] = Σⱼ A[i,j] * B[j,k]
@tullio C[i,k] := A[i,j] * B[j,k]

# Automatic GPU support, SIMD, threading
```

Fast, flexible, GPU-aware array operations.

### TensorOperations.jl
```julia
using TensorOperations

@tensor C[i,k] := A[i,j] * B[j,k]
```

Efficient tensor contractions.

## Statistics and Data

### Distributions.jl
```julia
using Distributions

d = Normal(0, 1)
x = rand(d, 1000)
pdf(d, 0.5)
cdf(d, 0.5)
```

### StatsBase.jl
Basic statistics functions.

### DataFrames.jl
Tabular data manipulation (like pandas).

## Plotting

### Plots.jl - High-Level Interface
```julia
using Plots

plot(x, y, label="data")
scatter!(x, y2, label="scatter")
```

Multiple backends: GR, PlotlyJS, PythonPlot.

### Makie.jl - High-Performance Visualization
```julia
using CairoMakie  # or GLMakie for interactive

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, x, y)
scatter!(ax, x, y2)
```

Better for: Publications, 3D, large datasets, interactive.

## Package Management

### Standard Registry
```julia
using Pkg

Pkg.add("PackageName")
Pkg.update()
Pkg.status()
```

### Local Development
```julia
# Develop local package
Pkg.develop(path="/path/to/package")

# Use Revise.jl for auto-reloading
using Revise
```

## Key Ecosystem Principles

1. **Composability**: Packages work together (e.g., GPU arrays work with DiffEq, AD)
2. **Generic code**: Write once, works with many types (Float32/64, GPU, AD)
3. **Multiple dispatch**: Type-based method selection enables optimization
4. **No two-language problem**: Write everything in Julia
