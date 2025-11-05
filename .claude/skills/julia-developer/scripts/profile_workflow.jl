#!/usr/bin/env julia
# Standard profiling and optimization workflow for Julia code

using Profile
using ProfileView  # For GUI visualization (optional)
using BenchmarkTools

"""
    profile_function(f, args...; show_gui=false, output_file="profile.html")

Profile a function and display results.

# Arguments
- `f`: Function to profile
- `args...`: Arguments to pass to function
- `show_gui`: Show ProfileView GUI (requires display, default: false)
- `output_file`: HTML output file for profile (default: "profile.html")

# Example
```julia
function myfunction(n)
    sum = 0.0
    for i in 1:n
        sum += sqrt(i)
    end
    return sum
end

profile_function(myfunction, 1_000_000)
```
"""
function profile_function(f, args...; show_gui=false, output_file="profile.html")
    println("🔍 Profiling: ", f)
    println("="^60)

    # Clear any existing profile data
    Profile.clear()

    # Run once to compile
    println("Warming up (compilation)...")
    f(args...)

    # Profile
    println("Profiling...")
    @profile f(args...)

    # Print text report
    println("\n📊 Profile Results:")
    Profile.print(format=:flat, sortedby=:count)

    # Export to file
    open(output_file, "w") do io
        Profile.print(io, format=:flat, sortedby=:count)
    end
    println("\n✅ Profile saved to: ", output_file)

    # Show GUI if requested
    if show_gui
        try
            ProfileView.view()
            println("📈 ProfileView GUI opened")
        catch e
            println("⚠️  Could not open ProfileView GUI: ", e)
            println("   This requires a display environment")
        end
    end

    return nothing
end

"""
    benchmark_function(f, args...; samples=1000)

Benchmark a function and display detailed timing statistics.

# Example
```julia
benchmark_function(myfunction, 1_000_000)
```
"""
function benchmark_function(f, args...; samples=1000)
    println("⏱️  Benchmarking: ", f)
    println("="^60)

    result = @benchmark $f($(args)...) samples=samples

    println("\n📊 Benchmark Results:")
    println(result)

    # Check for type instabilities
    println("\n🔍 Type Stability Check:")
    println("Run @code_warntype to check for type instabilities:")
    println("  @code_warntype $f($(join(args, ", ")))")

    return result
end

"""
    check_allocations(f, args...)

Check for unnecessary memory allocations.

# Example
```julia
check_allocations(myfunction, 1_000_000)
```
"""
function check_allocations(f, args...)
    println("💾 Checking allocations: ", f)
    println("="^60)

    # Run once to compile
    f(args...)

    # Check allocations
    allocs = @allocated f(args...)
    println("\nTotal allocations: ", allocs, " bytes")

    if allocs == 0
        println("✅ No allocations! Excellent!")
    elseif allocs < 1000
        println("✅ Minimal allocations (< 1KB)")
    else
        println("⚠️  Consider reducing allocations")
        println("   Use @time to see allocation count")
        println("   Use --track-allocation=user to find sources")
    end

    # Show timing with allocation details
    println("\n⏱️  Detailed timing:")
    @time f(args...)

    return allocs
end

"""
    optimization_checklist()

Print a checklist for optimizing Julia code.
"""
function optimization_checklist()
    println("""
    📋 Julia Optimization Checklist:
    ================================

    1. Type Stability
       □ Run @code_warntype to check for type instabilities
       □ Avoid abstract types in containers (use concrete types)
       □ Use type annotations for struct fields

    2. Memory Allocations
       □ Run @time to check allocation count
       □ Pre-allocate output arrays when possible
       □ Use in-place operations (!, e.g., mul! instead of *)
       □ Consider StaticArrays for small fixed-size arrays

    3. Performance Patterns
       □ Avoid global variables (or use const)
       □ Use @inbounds when you've verified bounds
       □ Use @simd for vectorizable loops
       □ Consider @views to avoid array copies
       □ Use @fastmath when approximate math is acceptable

    4. Multiple Dispatch
       □ Write type-specific methods for hot paths
       □ Avoid Union types in performance-critical code
       □ Use parametric types effectively

    5. Profiling
       □ Profile with real workload, not toy examples
       □ Focus on hotspots (>1% of runtime)
       □ Re-profile after each optimization

    6. GPU Computing
       □ Use CUDA.jl, Metal.jl, or KernelAbstractions.jl
       □ Ensure data types match GPU precision
       □ Minimize CPU-GPU transfers
       □ Use GPU-friendly algorithms (parallel, minimal branching)

    7. Benchmarking
       □ Use BenchmarkTools.jl (@benchmark, @btime)
       □ Compare against baseline implementations
       □ Test with realistic input sizes
       □ Watch for regressions

    Run this script's functions to profile, benchmark, and check allocations!
    """)
end

# Command-line interface
if !isempty(ARGS) && ARGS[1] == "--checklist"
    optimization_checklist()
else
    println("""
    Julia Profile Workflow
    ======================

    This script provides utilities for profiling and optimizing Julia code.

    Functions:
      - profile_function(f, args...)     Profile a function
      - benchmark_function(f, args...)   Benchmark a function
      - check_allocations(f, args...)    Check memory allocations
      - optimization_checklist()         Show optimization checklist

    Usage:
      1. Include this script: include("profile_workflow.jl")
      2. Call the profiling functions on your code
      3. Or run: julia profile_workflow.jl --checklist

    Example:
      ```julia
      include("profile_workflow.jl")

      function myfunction(n)
          sum(sqrt(i) for i in 1:n)
      end

      profile_function(myfunction, 1_000_000)
      benchmark_function(myfunction, 1_000_000)
      check_allocations(myfunction, 1_000_000)
      ```
    """)
end
