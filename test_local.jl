#!/usr/bin/env julia
#
# Local test script for MFGnet.jl
# This script helps test the package locally before pushing to CI
#
# Usage:
#   julia test_local.jl
#   julia test_local.jl --verbose
#   julia test_local.jl --test testNN
#

using Pkg

println("=" ^ 60)
println("MFGnet.jl Local Test Script")
println("=" ^ 60)

# Parse command line arguments
verbose = "--verbose" in ARGS
specific_test = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--test" && i < length(ARGS)
        specific_test = ARGS[i+1]
    end
end

# Activate the project
println("\n[1/5] Activating project...")
Pkg.activate(".")

# Instantiate dependencies
println("\n[2/5] Installing dependencies...")
try
    Pkg.instantiate()
    println("✓ Dependencies installed successfully")
catch e
    println("✗ Error installing dependencies:")
    println(e)
    exit(1)
end

# Precompile
println("\n[3/5] Precompiling...")
try
    Pkg.precompile()
    println("✓ Precompilation successful")
catch e
    println("⚠ Warning during precompilation:")
    println(e)
end

# Load the package
println("\n[4/5] Loading MFGnet...")
try
    using MFGnet
    println("✓ MFGnet loaded successfully")
catch e
    println("✗ Error loading MFGnet:")
    println(e)
    exit(1)
end

# Run tests
println("\n[5/5] Running tests...")
println("-" ^ 60)

if specific_test !== nothing
    println("Running specific test: $specific_test")
    include("test/$specific_test.jl")
else
    println("Running all tests...")
    try
        Pkg.test(; coverage=false)
        println("-" ^ 60)
        println("✓ All tests passed!")
    catch e
        println("-" ^ 60)
        println("✗ Tests failed:")
        println(e)
        exit(1)
    end
end

println("\n" * "=" ^ 60)
println("Testing complete!")
println("=" ^ 60)
