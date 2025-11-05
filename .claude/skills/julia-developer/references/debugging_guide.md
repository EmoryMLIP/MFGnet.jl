# Julia Debugging Guide

Comprehensive guide to debugging Julia code using various tools and techniques.

## Interactive Debugging with Debugger.jl

### Basic Usage

```julia
using Debugger

function problematic_function(x)
    y = x * 2
    z = y + 10
    return z / (x - 5)  # Problem when x == 5
end

# Start debugger
@enter problematic_function(5)
```

### Debugger Commands

```
n  - next line
s  - step into function call
c  - continue execution
finish - finish current function
u  - move up stack frame
d  - move down stack frame
q  - quit debugger

fr [n]  - show stack frame n
bt      - backtrace (show call stack)

`var    - evaluate variable
@eval expr - evaluate expression

bp add file:line - add breakpoint
bp rm n - remove breakpoint
bp - list breakpoints
```

### Example Session

```julia
julia> @enter problematic_function(5)
In problematic_function(x) at REPL[1]:1
   1  function problematic_function(x)
>  2      y = x * 2
   3      z = y + 10
   4      return z / (x - 5)
   5  end

1|debug> n
In problematic_function(x) at REPL[1]:2
   2      y = x * 2
>  3      z = y + 10
   4      return z / (x - 5)

1|debug> `y
10

1|debug> `x
5

1|debug> n
In problematic_function(x) at REPL[1]:3
   3      z = y + 10
>  4      return z / (x - 5)

1|debug> @eval x - 5
0

1|debug> # Ah, division by zero!
```

## Lightweight Debugging with Infiltrator.jl

Fast breakpoints without full debugger:

```julia
using Infiltrator

function my_function(x)
    y = complex_computation(x)

    @infiltrate  # Breakpoint here

    z = another_computation(y)
    return z
end

my_function(10)
```

When execution hits `@infiltrate`:
- Pauses execution
- Drops into REPL
- Can inspect all variables
- Type `@continue` to resume

### Conditional Breakpoints

```julia
@infiltrate x > 100  # Only break if condition true
```

### Infiltrator Commands

```julia
# In infiltrator mode:
@locals        # Show all local variables
@exfiltrate    # Save variables to Main for later inspection
@continue      # Resume execution
@doc symbol    # Show documentation
```

## VS Code Debugger

### Setup

1. Install Julia VS Code extension
2. Open Julia file
3. Set breakpoints (click left margin)
4. Press F5 or Run → Start Debugging

### Features

- Visual breakpoints
- Step through code (F10/F11)
- Variable inspector
- Watch expressions
- Call stack navigation
- Conditional breakpoints

### Launch Configuration (.vscode/launch.json)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "julia",
            "request": "launch",
            "name": "Run active Julia file",
            "program": "${file}",
            "stopOnEntry": false,
            "cwd": "${workspaceFolder}",
            "juliaEnv": "${command:activeJuliaEnvironment}"
        }
    ]
}
```

## Logging and Assertions

### @info, @warn, @error

```julia
using Logging

function process_data(data)
    @info "Processing data" size=length(data)

    if length(data) == 0
        @warn "Empty data received"
        return nothing
    end

    result = compute(data)

    @debug "Intermediate result" result=result
    @error "Something went wrong" exception=(err, catch_backtrace())

    return result
end

# Set logging level
Logging.global_logger(ConsoleLogger(stderr, Logging.Debug))
```

### @assert

```julia
function my_function(x)
    @assert x > 0 "x must be positive"
    @assert length(x) > 10 "Need at least 10 elements"

    # Implementation
end
```

Assertions are expensive; remove for production using:
```julia
julia --optimize=2  # Removes @assert
```

## Printf-Style Debugging

```julia
function debug_function(x)
    println("Entering with x = ", x)

    y = compute(x)
    println("After compute: y = ", y)

    @show y  # Prints "y = value"
    @show typeof(y)

    return y
end

# Pretty printing
using PrettyTables
pretty_table(data)
```

## Introspection Tools

### @code_* Macros

```julia
# What code is actually being run?
@code_lowered my_function(x)  # Lowered IR
@code_typed my_function(x)    # Type-inferred IR
@code_llvm my_function(x)     # LLVM IR
@code_native my_function(x)   # Assembly

# Which method is called?
@which my_function(x)

# Show all methods
methods(my_function)
```

### Type Inspection

```julia
@code_warntype my_function(x)  # Check type stability

# Identify type instabilities (look for red Union or yellow Any)
```

Example:
```julia
function unstable(x)
    if x > 0
        return x
    else
        return 0.0
    end
end

@code_warntype unstable(5)
# Shows Union{Int64, Float64} in red (BAD)
```

### Stack Traces

```julia
try
    error_prone_function()
catch e
    @error "Error occurred" exception=(e, catch_backtrace())
    showerror(stdout, e, catch_backtrace())
end
```

## Performance Debugging

### Profiling

```julia
using Profile

# Profile function
@profile my_function(x)

# Show results
Profile.print()

# Visual profiling (requires display)
using ProfileView
@profview my_function(x)
```

### Allocation Profiling

```julia
# Check allocations
@time my_function(x)
@allocated my_function(x)

# Detailed allocation tracking
julia --track-allocation=user
# Run code
# Exit julia
# Check *.mem files
```

### Benchmarking

```julia
using BenchmarkTools

@btime my_function($x)  # Quick timing

@benchmark my_function($x)  # Detailed stats
```

## Debugging Specific Issues

### MethodError

```julia
# ERROR: MethodError: no method matching foo(::String)

# Solutions:
methods(foo)  # See what methods exist
@which foo(1)  # See which method would be called

# Add missing method or fix argument types
```

### UndefVarError

```julia
# ERROR: UndefVarError: x not defined

# Check scope:
function bad()
    if rand() > 0.5
        x = 10
    end
    return x  # x might not be defined!
end

# Fix:
function good()
    x = 0  # Define in outer scope
    if rand() > 0.5
        x = 10
    end
    return x
end
```

### Type Instability

```julia
# Diagnose with @code_warntype
@code_warntype problematic_function(x)

# Common causes:
# 1. Global variables
# 2. Changing variable types
# 3. Untyped containers
# 4. Abstract field types in structs
```

### Stack Overflow

```julia
# Recursion too deep

# Solutions:
# 1. Increase stack size: julia --stack-size=10M
# 2. Convert to iterative
# 3. Use tail recursion (Julia doesn't optimize TCO)
```

### Memory Leaks

```julia
# Check memory usage
using InteractiveUtils
varinfo()  # Show memory usage of variables

# Explicit garbage collection
GC.gc(true)

# Profile memory
using Profile
Profile.Allocs.@profile sample_rate=0.0001 my_function(x)
```

## Debugging GPU Code

### CUDA Debugging

```julia
using CUDA

# Check CUDA errors immediately
CUDA.@sync @cuda kernel!(y, x)

# Allow scalar indexing for debugging (SLOW!)
CUDA.allowscalar() do
    println(x_gpu[1])
end

# Memory debugging
CUDA.memory_status()

# Reclaim memory
GC.gc(true)
CUDA.reclaim()
```

### GPU Kernel Debugging

```julia
# Add print statements in kernel (limited)
function debug_kernel!(y, x)
    i = threadIdx().x
    if i == 1
        @cuprintln("First thread, x[1] = ", x[1])
    end
    @inbounds y[i] = x[i] * 2
    return nothing
end

@cuda threads=256 blocks=1 debug_kernel!(y, x)
CUDA.synchronize()
```

## Debugging Packages

### Package Development

```julia
# Develop package locally
] dev path/to/MyPackage

# Use Revise for auto-reloading
using Revise
using MyPackage

# Edit code, changes automatically reload

# Debug package tests
] test MyPackage

# Or manually
using MyPackage
include("test/runtests.jl")
```

### Debugging Dependencies

```julia
# Check what's loaded
@which problematic_function(x)

# Check package versions
] status

# Update packages
] update

# Resolve dependency conflicts
] resolve
```

## Common Debugging Patterns

### Binary Search Debugging

```julia
function complex_function(x)
    # Checkpoint 1
    @info "Checkpoint 1" x=x

    step1 = operation1(x)
    # Checkpoint 2
    @info "Checkpoint 2" step1=step1

    step2 = operation2(step1)
    # Checkpoint 3
    @info "Checkpoint 3" step2=step2

    return step3
end
```

Narrow down where error occurs by adding checkpoints.

### Minimal Working Example (MWE)

When debugging complex code:

1. Copy problematic code
2. Remove unrelated parts
3. Simplify inputs
4. Reproduce error with minimal code

```julia
# Original: 1000 lines
# MWE: 10 lines that reproduce bug

function mwe()
    x = [1, 2, 3]
    result = problematic_operation(x)
    return result
end
```

### Isolation Testing

```julia
# Test components separately

@testset "Component isolation" begin
    # Test step 1
    result1 = step1(input)
    @test result1 isa ExpectedType

    # Test step 2 with known input
    result2 = step2(known_good_input)
    @test result2 ≈ expected_output

    # Test full pipeline
    final = step3(step2(step1(input)))
end
```

## Debugging Tips

### 1. Read Error Messages Carefully

```julia
# ERROR: DimensionMismatch("dimensions must match: a has dims (3,), b has dims (4,), mismatch at 1")

# Error tells you exactly what's wrong!
```

### 2. Check Types

```julia
@show typeof(x)
@show size(x)
@show eltype(x)
```

### 3. Simplify

- Remove complexity until error disappears
- Add back one piece at a time
- Find what causes the problem

### 4. Test Assumptions

```julia
# Assumption: x is always positive
@assert all(x .> 0)  # Test it!

# Assumption: array not empty
@assert !isempty(x)
```

### 5. Use the REPL

- Interactive exploration
- Test small pieces of code
- Inspect variables directly

### 6. Bisect History

If code worked before:
```bash
git bisect start
git bisect bad  # Current (broken) commit
git bisect good v1.0.0  # Last known good version
# Git will check out middle commit, test it
git bisect good/bad  # Based on test
# Repeat until bug found
```

## Debugging Checklist

When stuck:
- [ ] Read error message completely
- [ ] Check types with `@show typeof(...)`
- [ ] Use `@code_warntype` for type stability
- [ ] Add `@infiltrate` or `@enter` to pause execution
- [ ] Simplify to minimal working example
- [ ] Test components in isolation
- [ ] Check assumptions with `@assert`
- [ ] Profile if performance issue
- [ ] Check package versions `] status`
- [ ] Search GitHub issues
- [ ] Ask on Discourse/Slack with MWE

## VS Code Tips

**Keyboard shortcuts:**
- F5: Start debugging
- F9: Toggle breakpoint
- F10: Step over
- F11: Step into
- Shift+F11: Step out
- F5: Continue

**Debug console:**
- Evaluate expressions
- Inspect variables
- Call functions

## Summary

**Interactive debugging:**
- Debugger.jl: Full-featured debugger
- Infiltrator.jl: Fast breakpoints
- VS Code: Visual debugging

**Logging:**
- `@info`, `@warn`, `@error` for runtime logging
- `@debug` for detailed logging
- `@assert` for invariant checking

**Introspection:**
- `@code_warntype`: Type stability
- `@code_*`: See generated code
- `@which`, `methods`: Method dispatch

**Performance:**
- Profile.jl: Time profiling
- `@time`, `@allocated`: Allocation checking
- BenchmarkTools.jl: Accurate benchmarking

**Best practices:**
- Read error messages
- Simplify to MWE
- Test assumptions
- Use REPL for exploration
- Isolate components
