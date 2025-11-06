"""
Main test suite for MFGnet.jl migration

Runs comprehensive tests for Julia ecosystem migration:
- Unit tests for new components
- Integration tests comparing legacy vs new
- Validation tests for gradients and convergence
- End-to-end training tests
- Backward compatibility tests

Usage:
    julia --project=. test/runtests_migration.jl
    julia --project=. test/runtests_migration.jl --quick     # Fast smoke test
    julia --project=. test/runtests_migration.jl --unit      # Unit tests only
    julia --project=. test/runtests_migration.jl --full      # Full test suite
"""

using Test
using LinearAlgebra
using Printf
using MFGnet

println("="^70)
println("MFGnet.jl Migration Test Suite")
println("="^70)
println("Julia version: $(VERSION)")
println("MFGnet version: $(pkgversion(MFGnet))")
println("="^70)

# Parse command line arguments
test_mode = length(ARGS) > 0 ? ARGS[1] : "--full"

if test_mode == "--quick"
    println("Running QUICK tests (smoke tests only)...")
    test_sets = [:unit_quick]
elseif test_mode == "--unit"
    println("Running UNIT tests...")
    test_sets = [:unit]
elseif test_mode == "--full"
    println("Running FULL test suite...")
    test_sets = [:unit, :integration, :validation, :e2e, :compat]
else
    error("Unknown test mode: $test_mode. Use --quick, --unit, or --full")
end

# Track timing
test_times = Dict{Symbol, Float64}()

# ============================================================================
# Utility Tests (Always Run)
# ============================================================================

@testset "Test Utilities" begin
    println("\n" * "-"^70)
    println("Testing utility functions...")
    println("-"^70)

    include("utils/test_utils.jl")
    include("utils/test_problems.jl")

    using .TestUtils
    using .TestProblems

    @testset "TestUtils module" begin
        # Test vec_params and reconstruct_params
        Θ = ((randn(3,2), randn(3)), (randn(2,3), randn(2)))
        v = vec_params(Θ)
        Θ_recon = reconstruct_params(v, Θ)

        @test vec_params(Θ_recon) ≈ v
    end

    @testset "TestProblems module" begin
        # Test problem creation
        prob = gaussian_1d_transport(nex=10)
        @test haskey(prob, :mfg)
        @test haskey(prob, :Θ_init)
        @test haskey(prob, :properties)

        # List problems
        list_test_problems()
    end
end

# ============================================================================
# Unit Tests
# ============================================================================

if :unit in test_sets || :unit_quick in test_sets
    t_start = time()

    @testset "Unit Tests" begin
        println("\n" * "="^70)
        println("UNIT TESTS")
        println("="^70)

        if :unit_quick in test_sets
            # Quick smoke tests only
            @testset "Quick Smoke Tests" begin
                println("Testing basic MFG problem creation...")
                prob = TestProblems.gaussian_1d_transport(nex=10)
                mfg, Θ = prob.mfg, prob.Θ_init

                # Can evaluate
                Jc = mfg(Θ)
                @test isfinite(Jc)

                # Can compute gradient
                using Zygote
                ∇Jc = Zygote.gradient(θ -> mfg(θ), Θ)[1]
                @test all(isfinite.(TestUtils.vec_params(∇Jc)))

                println("✓ Basic functionality works")
            end
        else
            # Full unit tests
            @testset "ODE Wrapper" begin
                println("\nTesting ODE wrapper...")
                include("unit/test_ode_wrapper.jl")
            end

            # NOTE: These would be added during migration
            # @testset "Optimization Wrapper" begin
            #     println("\nTesting optimization wrapper...")
            #     include("unit/test_optimization_wrapper.jl")
            # end

            # @testset "ComponentArrays" begin
            #     println("\nTesting ComponentArray conversion...")
            #     include("unit/test_componentarrays.jl")
            # end
        end
    end

    test_times[:unit] = time() - t_start
end

# ============================================================================
# Integration Tests
# ============================================================================

if :integration in test_sets
    t_start = time()

    @testset "Integration Tests" begin
        println("\n" * "="^70)
        println("INTEGRATION TESTS")
        println("="^70)

        @testset "Backward Compatibility" begin
            println("\nTesting backward compatibility...")
            include("integration/test_backward_compat.jl")
        end

        # NOTE: Would add during migration
        # @testset "Numerical Equivalence" begin
        #     println("\nTesting numerical equivalence (legacy vs new)...")
        #     include("integration/test_numerical_equivalence.jl")
        # end
    end

    test_times[:integration] = time() - t_start
end

# ============================================================================
# Validation Tests
# ============================================================================

if :validation in test_sets
    t_start = time()

    @testset "Validation Tests" begin
        println("\n" * "="^70)
        println("VALIDATION TESTS")
        println("="^70)

        @testset "Gradient Validation" begin
            println("\nTesting gradient correctness...")
            include("validation/test_gradients.jl")
        end

        # NOTE: Would add during migration
        # @testset "Convergence Properties" begin
        #     println("\nTesting convergence...")
        #     include("validation/test_convergence.jl")
        # end

        # @testset "Conservation Laws" begin
        #     println("\nTesting conservation properties...")
        #     include("validation/test_conservation.jl")
        # end
    end

    test_times[:validation] = time() - t_start
end

# ============================================================================
# End-to-End Tests
# ============================================================================

if :e2e in test_sets
    t_start = time()

    @testset "End-to-End Tests" begin
        println("\n" * "="^70)
        println("END-TO-END TESTS")
        println("="^70)

        @testset "Full Training Pipeline" begin
            println("\nTesting complete training...")
            include("e2e/test_full_training.jl")
        end

        # NOTE: Would add during migration
        # @testset "Performance Benchmarks" begin
        #     println("\nRunning benchmarks...")
        #     include("e2e/test_benchmarks.jl")
        # end
    end

    test_times[:e2e] = time() - t_start
end

# ============================================================================
# Compatibility with Original Tests
# ============================================================================

if :compat in test_sets
    @testset "Original Test Suite" begin
        println("\n" * "="^70)
        println("ORIGINAL TEST SUITE")
        println("="^70)
        println("Running original tests to ensure no regression...")

        # Run original tests
        @testset "NN" begin
            include("testNN.jl")
        end

        @testset "param2vec" begin
            include("testParam2Vec.jl")
        end

        @testset "PotentialNN" begin
            include("testPotentialNN.jl")
        end

        @testset "PotentialResNN" begin
            include("testPotentialResNN.jl")
        end

        @testset "PotentialSingle" begin
            include("testPotentialSingle.jl")
        end

        @testset "ResNN" begin
            include("testResNN.jl")
        end

        @testset "singleLayer" begin
            include("testSingleLayer.jl")
        end

        @testset "linInter1D" begin
            include("testLinInter1D.jl")
        end
    end
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("TEST SUMMARY")
println("="^70)

if !isempty(test_times)
    println("\nTest Timings:")
    for (name, t) in test_times
        @printf("  %-20s: %.2f seconds\n", string(name), t)
    end
    total_time = sum(values(test_times))
    @printf("\n  Total time: %.2f seconds\n", total_time)
end

println("\n" * "="^70)
println("All tests completed!")
println("="^70)
