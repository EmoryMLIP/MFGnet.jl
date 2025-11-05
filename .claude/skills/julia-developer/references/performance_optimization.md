# Julia Performance Optimization Guide

Comprehensive guide to writing high-performance Julia code leveraging the JIT compiler.

## Type Stability

Type stability is the **most important** performance concept in Julia.

### What is Type Stability?

A function is type-stable if the type of the output can be inferred from the types of the inputs (without running the code).

**Type-unstable (BAD):**
```julia
function unstable(x)
    if x > 0
        return x          # Returns Int or Float
    else
        return 0.0        # Returns Float64
    end
end
```

**Type-stable (GOOD):**
```julia
function stable(x)
    if x > 0
        return Float64(x)  # Always returns Float64
    else
        return 0.0
    end
end
```

### Checking Type Stability

Use `@code_warntype`:

```julia
@code_warntype unstable(5)   # Shows type instability in red
@code_warntype stable(5)     # All types inferred correctly
```

Look for:
- **Red text** (e.g., `Union{Int64, Float64}`): Type instability
- **Yellow `Any`**: Compiler gave up on type inference
- **Blue concrete types** (e.g., `Float64`): Good!

### Common Type Stability Issues

#### 1. Untyped containers

**BAD:**
```julia
function sum_array()
    arr = []  # Type is Vector{Any}
    for i in 1:100
        push!(arr, i)
    end
    sum(arr)
end
```

**GOOD:**
```julia
function sum_array()
    arr = Int[]  # Type is Vector{Int}
    for i in 1:100
        push!(arr, i)
    end
    sum(arr)
end
```

#### 2. Abstract field types

**BAD:**
```julia
struct BadContainer
    data::AbstractArray  # Abstract type
end
```

**GOOD:**
```julia
struct GoodContainer{T<:AbstractArray}
    data::T  # Concrete type at instantiation
end
```

#### 3. Global variables

**BAD:**
```julia
x = 5

function use_global()
    return x + 1  # Compiler can't infer type of x
end
```

**GOOD:**
```julia
const x = 5  # Constant global

function use_global()
    return x + 1  # Type inference works
end
```

Or pass as argument:
```julia
function use_argument(x)
    return x + 1
end
```

#### 4. Changing variable types

**BAD:**
```julia
function bad_loop()
    x = 0        # x is Int
    for i in 1:10
        x = x + 0.5  # Now x is Float64!
    end
    return x
end
```

**GOOD:**
```julia
function good_loop()
    x = 0.0      # x is Float64 from start
    for i in 1:10
        x = x + 0.5
    end
    return x
end
```

## Memory Allocations

Reducing allocations is the second most important optimization.

### Measuring Allocations

```julia
@time f(x)        # Shows time and allocations
@allocated f(x)   # Shows only allocation count (bytes)
```

### Pre-allocation

**BAD (allocates every iteration):**
```julia
function bad_loop(n)
    for i in 1:n
        x = zeros(1000)  # Allocates!
        # ... use x
    end
end
```

**GOOD (pre-allocate):**
```julia
function good_loop(n)
    x = zeros(1000)    # Allocate once
    for i in 1:n
        fill!(x, 0.0)   # Reuse
        # ... use x
    end
end
```

### In-Place Operations

Julia uses `!` convention for functions that mutate arguments.

**BAD (allocates):**
```julia
C = A * B          # Allocates new matrix C
```

**GOOD (in-place):**
```julia
using LinearAlgebra
mul!(C, A, B)      # Writes into pre-allocated C
```

Common in-place functions:
- `mul!(C, A, B)`: Matrix multiply
- `copy!(dst, src)`: Copy array
- `fill!(x, val)`: Fill array
- `sort!(x)`: Sort in-place
- `map!(f, dst, src)`: Apply function

### Views Instead of Copies

**BAD (copies):**
```julia
function sum_first_col(A)
    col = A[:, 1]      # Copies column
    return sum(col)
end
```

**GOOD (view):**
```julia
function sum_first_col(A)
    col = @view A[:, 1]  # No copy, just a view
    return sum(col)
end
```

Or use `@views` macro:
```julia
@views function sum_first_col(A)
    col = A[:, 1]      # Automatically a view
    return sum(col)
end
```

### StaticArrays for Small Arrays

For small fixed-size arrays (<100 elements), use StaticArrays:

```julia
using StaticArrays

# Stack-allocated, no heap allocation
v = SVector(1, 2, 3)
m = SMatrix{3,3}(1,2,3,4,5,6,7,8,9)

# Operations don't allocate
w = v + v  # No allocation!
```

**When to use:**
- Small arrays (typically < 100 elements)
- Size known at compile time
- Performance-critical inner loops

## Loop Optimization

### @inbounds - Skip Bounds Checking

```julia
function sum_array(x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += x[i]  # No bounds check
    end
    return s
end
```

**Warning:** Only use when you're certain bounds are valid. Undefined behavior otherwise.

### @simd - SIMD Vectorization

```julia
function sum_array(x)
    s = zero(eltype(x))
    @simd for i in eachindex(x)
        s += x[i]  # May use SIMD instructions
    end
    return s
end
```

Requirements for `@simd`:
- Simple loop body
- No loop-carried dependencies
- Iterations independent

### @fastmath - Fast Math

```julia
@fastmath begin
    y = sqrt(x^2 + y^2)  # Fast but less accurate
end
```

**Warning:** Breaks IEEE floating-point semantics. Use only when:
- Approximate results acceptable
- Performance critical
- You understand the trade-offs

### Loop Fusion

**BAD (multiple passes):**
```julia
y = sqrt.(x)
z = exp.(y)
w = log.(z)
```

**GOOD (single pass):**
```julia
w = map(xi -> log(exp(sqrt(xi))), x)
```

Or use `@.` for vectorized operations:
```julia
@. w = log(exp(sqrt(x)))
```

## Function Barriers

When type instability is unavoidable, use function barriers.

**BAD:**
```julia
function process_data(data)
    # Type of result unknown
    result = complex_parsing(data)

    # Compiler can't optimize this
    for i in 1:length(result)
        # ... expensive operations
    end
end
```

**GOOD:**
```julia
function process_data(data)
    result = complex_parsing(data)  # Type-unstable
    process_result(result)           # Function barrier
end

function process_result(result)
    # Type of result is known here
    for i in 1:length(result)
        # Compiler can optimize
    end
end
```

The inner function `process_result` has stable types even though the outer doesn't.

## Struct Design

### Immutable When Possible

```julia
struct Point2D  # Immutable by default
    x::Float64
    y::Float64
end
```

**Benefits:**
- Stack-allocated when small
- Better optimization
- Thread-safe

Use `mutable struct` only when necessary.

### Parametric Types

**BAD:**
```julia
struct BadContainer
    data::AbstractVector  # Abstract type
end
```

**GOOD:**
```julia
struct GoodContainer{T<:AbstractVector}
    data::T  # Concrete type parameter
end
```

Compiler generates specialized code for each `T`.

### Avoid Small Unions

```julia
# OK: Small union (2-3 types)
struct Result
    value::Union{Float64, Nothing}
end

# BAD: Large union
struct BadResult
    value::Union{Int64, Float64, String, Nothing, Missing}
end
```

Small unions (2-3 types) are optimized; larger unions are not.

## Profiling

### Profile.jl

```julia
using Profile

@profile my_function(x)
Profile.print()
```

Shows where time is spent. Focus on:
- Functions with high sample count
- Unexpected allocations
- Type instabilities

### ProfileView.jl

```julia
using ProfileView

@profview my_function(x)
```

Visual flame graph of performance.

### Allocation Profiling

```julia
julia --track-allocation=user
# Run your code
# Exit Julia
# Check *.mem files
```

Shows allocation sites line-by-line.

## Benchmarking

### BenchmarkTools.jl

```julia
using BenchmarkTools

# Single run with timing
@btime my_function($x)  # $ for interpolation

# Detailed statistics
@benchmark my_function($x)
```

**Always use `$` interpolation** to avoid benchmarking variable lookup.

### Comparing Performance

```julia
function compare_implementations()
    x = rand(1000)

    println("Method 1:")
    @btime method1($x)

    println("Method 2:")
    @btime method2($x)
end
```

## Common Performance Patterns

### 1. Type-Specialized Dispatch

```julia
# Generic fallback (slower)
process(x::AbstractArray) = sum(x)

# Specialized for concrete types (faster)
process(x::Vector{Float64}) = # optimized implementation
process(x::CuArray{Float32}) = # GPU implementation
```

### 2. Generated Functions

For compile-time computation:

```julia
@generated function my_function(x::Array{T,N}) where {T,N}
    # Code here runs at compile time
    # Can specialize on N, T, etc.
    return quote
        # Generated code here
    end
end
```

### 3. Loop Unrolling with Meta-programming

```julia
function unrolled_sum(x::SVector{N}) where N
    s = zero(eltype(x))
    Base.Cartesian.@nexprs $N i -> s += x[i]
    return s
end
```

## GPU-Specific Optimization

### Use Appropriate Precision

```julia
# GPU is often faster with Float32
x = CuArray{Float32}(x_cpu)
```

### Minimize Transfers

```julia
# BAD: Multiple transfers
y = Array(x_gpu)  # GPU -> CPU
z = process(y)
w_gpu = CuArray(z)  # CPU -> GPU

# GOOD: Keep on GPU
w_gpu = gpu_process(x_gpu)
```

### Kernel Fusion

```julia
# BAD: Multiple kernel launches
y = sqrt.(x)
z = exp.(y)

# GOOD: Fused kernel
z = exp.(sqrt.(x))  # Single kernel
```

## Performance Checklist

Before optimization:
- [ ] Profile to find bottlenecks
- [ ] Use `@time` to check allocations
- [ ] Run `@code_warntype` on hot functions

Core optimizations:
- [ ] Ensure type stability
- [ ] Pre-allocate arrays
- [ ] Use in-place operations (`!` functions)
- [ ] Use `@views` to avoid copies
- [ ] Consider `StaticArrays` for small arrays
- [ ] Avoid global variables (or use `const`)

Advanced optimizations:
- [ ] Use `@inbounds` when safe
- [ ] Use `@simd` for vectorizable loops
- [ ] Consider `@fastmath` if accuracy allows
- [ ] Write type-specialized methods
- [ ] Use function barriers for unavoidable type instability

After optimization:
- [ ] Re-profile to verify improvement
- [ ] Benchmark against baseline
- [ ] Test correctness thoroughly

## When NOT to Optimize

1. **Premature optimization**: Profile first
2. **Non-bottlenecks**: Don't optimize code that's < 1% of runtime
3. **One-time code**: If it runs once, readability > speed
4. **Correct first**: Make it work, then make it fast
