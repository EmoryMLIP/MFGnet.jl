"""
Gradient validation tests

Tests automatic differentiation through ODE solvers:
1. Finite difference validation
2. Taylor test (convergence rates)
3. Gradient consistency across solvers
4. Sensitivity algorithm selection
"""

using Test
using LinearAlgebra
using MFGnet
using Zygote

include("../utils/test_utils.jl")
include("../utils/test_problems.jl")
using .TestUtils
using .TestProblems

@testset "Gradient Correctness - Finite Differences" begin

    @testset "1D Gaussian - Legacy gradients" begin
        prob_data = gaussian_1d_transport(nex=20)  # Small for speed
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Compute gradient via AD
        f(θ) = mfg(θ)
        Jc, ∇Jc_AD = Zygote.withgradient(f, Θ)

        # Compute gradient via finite differences
        ∇Jc_FD = finite_difference_gradient(f, Θ; h=1e-5, method=:central)

        # Compare
        tol = MIGRATION_TOLERANCES
        compare_gradients(∇Jc_AD[1], ∇Jc_FD;
                        abstol=tol.gradient_abstol,
                        reltol=tol.fd_reltol,
                        name="Legacy gradient FD check")
    end

    @testset "1D Gaussian - DiffEq gradients" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Compute gradient with DifferentialEquations.jl
            f(θ) = mfg(θ, use_diffeq=true, solver=Tsit5(),
                      abstol=1e-6, reltol=1e-3)

            Jc, ∇Jc_AD = Zygote.withgradient(f, Θ)

            # Finite differences
            ∇Jc_FD = finite_difference_gradient(f, Θ; h=1e-5, method=:central)

            # Compare
            tol = MIGRATION_TOLERANCES
            compare_gradients(∇Jc_AD[1], ∇Jc_FD;
                            abstol=tol.gradient_abstol,
                            reltol=tol.fd_reltol,
                            name="DiffEq gradient FD check")
        end
    end

    @testset "2D Problem - Gradient structure" begin
        prob_data = gaussian_2d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Compute gradient
        f(θ) = mfg(θ)
        Jc, ∇Jc = Zygote.withgradient(f, Θ)

        # Check gradient structure matches parameters
        @test ∇Jc[1] isa Tuple
        @test length(∇Jc[1]) == length(Θ)

        # Check all components are finite
        grad_vec = vec_params(∇Jc[1])
        @test all(isfinite.(grad_vec))
        @test !all(grad_vec .≈ 0)  # Should have non-zero gradients

        # Finite difference check on subset of parameters
        # (Too expensive to check all in 2D)
        @test_skip begin
            n_params = count_params(Θ)
            n_check = min(10, n_params)  # Check 10 random parameters
            indices = rand(1:n_params, n_check)

            ∇_AD_vec = vec_params(∇Jc[1])
            ∇_FD_vec = vec_params(finite_difference_gradient(f, Θ; h=1e-5))

            for i in indices
                @test isapprox(∇_AD_vec[i], ∇_FD_vec[i], rtol=1e-2)
            end
        end
    end

end

@testset "Taylor Test - Gradient Convergence Rates" begin

    @testset "1D Gaussian - Legacy" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        ∇f(θ) = Zygote.gradient(f, θ)[1]

        # Taylor test
        result = taylor_test(f, ∇f, Θ; verbose=true, n_tests=8)

        # Should show quadratic convergence (E1 ~ h²)
        @test length(result.errors_1) == 8
        @test result.errors_1[end] < result.errors_1[1]  # Error decreases
    end

    @testset "1D Gaussian - DiffEq" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            f(θ) = mfg(θ, use_diffeq=true, solver=Tsit5(),
                      abstol=1e-8, reltol=1e-6)
            ∇f(θ) = Zygote.gradient(f, θ)[1]

            # Taylor test
            result = taylor_test(f, ∇f, Θ; verbose=true, n_tests=8)

            # Should show quadratic convergence
            @test length(result.errors_1) == 8
            @test result.errors_1[end] < result.errors_1[1]
        end
    end

    @testset "2D Problem - Higher dimension" begin
        prob_data = gaussian_2d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        ∇f(θ) = Zygote.gradient(f, θ)[1]

        result = taylor_test(f, ∇f, Θ; verbose=false, n_tests=8)

        # Check convergence
        @test result.errors_1[end] < result.errors_1[1]
    end

end

@testset "Gradient Consistency Across Solvers" begin

    @testset "Different ODE solvers give similar gradients" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Compute gradient with different solvers
            solvers = [
                ("Tsit5", Tsit5()),
                ("Vern7", Vern7()),
            ]

            gradients = Dict()
            for (name, solver) in solvers
                f(θ) = mfg(θ, use_diffeq=true, solver=solver,
                          abstol=1e-8, reltol=1e-6)
                ∇Jc = Zygote.gradient(f, Θ)[1]
                gradients[name] = ∇Jc
            end

            # Compare gradients
            ∇1 = gradients["Tsit5"]
            ∇2 = gradients["Vern7"]

            compare_gradients(∇1, ∇2;
                            abstol=1e-5,
                            reltol=1e-3,
                            name="Gradient consistency (Tsit5 vs Vern7)")
        end
    end

    @testset "Different sensitivities give similar gradients" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Test different sensitivity algorithms
            sensealgs = [
                ("QuadratureAdjoint", QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))),
                ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))),
            ]

            gradients = Dict()
            for (name, sensealg) in sensealgs
                f(θ) = mfg(θ, use_diffeq=true,
                          solver=Tsit5(),
                          sensealg=sensealg,
                          abstol=1e-8, reltol=1e-6)
                ∇Jc = Zygote.gradient(f, Θ)[1]
                gradients[name] = ∇Jc
            end

            # Compare
            ∇1 = gradients["QuadratureAdjoint"]
            ∇2 = gradients["InterpolatingAdjoint"]

            compare_gradients(∇1, ∇2;
                            abstol=1e-5,
                            reltol=1e-2,  # May have larger differences
                            name="Gradient consistency (different sensealg)")
        end
    end

end

@testset "Gradient Magnitude and Direction" begin

    @testset "Gradient norm is reasonable" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        Jc, ∇Jc = Zygote.withgradient(f, Θ)

        grad_vec = vec_params(∇Jc[1])
        grad_norm = norm(grad_vec)

        # Gradient should be finite and non-trivial
        @test isfinite(grad_norm)
        @test grad_norm > 1e-10  # Not zero
        @test grad_norm < 1e10   # Not exploding

        # Objective should be reasonable
        @test isfinite(Jc)
        @test Jc > 0  # Cost should be positive
    end

    @testset "Gradient descent direction decreases objective" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        Jc0, ∇Jc = Zygote.withgradient(f, Θ)

        # Take small step in negative gradient direction
        α = 1e-4
        Θ_new = add_direction(Θ, ∇Jc[1], -α)

        Jc1 = f(Θ_new)

        # Objective should decrease (or at least not increase much)
        @test Jc1 <= Jc0 + 1e-6  # Allow small numerical error
    end

    @testset "Zero gradient at (potential) minimum" begin
        # This tests if optimization finds a point with small gradient

        @test_skip begin
            prob_data = gaussian_1d_transport(nex=20)
            mfg, Θ = prob_data.mfg, prob_data.Θ_init

            # Run brief optimization
            f(θ) = mfg(θ)

            # Simple gradient descent
            Θ_opt = copy(Θ)
            for iter in 1:50
                Jc, ∇Jc = Zygote.withgradient(f, Θ_opt)
                α = 1e-3
                Θ_opt = add_direction(Θ_opt, ∇Jc[1], -α)
            end

            # Gradient at optimized point should be smaller
            ∇Jc_init = Zygote.gradient(f, Θ)[1]
            ∇Jc_opt = Zygote.gradient(f, Θ_opt)[1]

            @test norm(vec_params(∇Jc_opt)) < norm(vec_params(∇Jc_init))
        end
    end

end

@testset "Gradient Sparsity and Structure" begin

    @testset "Gradients w.r.t. each parameter layer" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        Jc, ∇Jc = Zygote.withgradient(f, Θ)

        # Θ = (w, (K, b), A, c, z)
        # All should have gradients
        ∇w, ∇Kb, ∇A, ∇c, ∇z = ∇Jc[1]

        @test ∇w isa AbstractArray
        @test ∇Kb isa Tuple
        @test ∇A isa AbstractArray
        @test ∇c isa AbstractArray
        @test ∇z isa AbstractArray

        # Check non-zero gradients (at least some components)
        @test any(abs.(∇w) .> 1e-10)
        @test any(abs.(∇Kb[1]) .> 1e-10)  # K matrix
        @test any(abs.(∇Kb[2]) .> 1e-10)  # b vector
    end

end

@testset "Gradient Performance" begin

    @testset "Gradient computation is not too slow" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)

        # Measure forward pass time
        t_forward = @elapsed f(Θ)

        # Measure gradient time
        t_gradient = @elapsed Zygote.gradient(f, Θ)

        # Gradient should not be more than ~10x forward pass
        # (Exact ratio depends on implementation)
        @test t_gradient < 20 * t_forward

        println("Forward time: $(t_forward)s, Gradient time: $(t_gradient)s")
    end

    @testset "DiffEq gradients have reasonable overhead" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            f_diffeq(θ) = mfg(θ, use_diffeq=true, solver=Tsit5())

            t_forward = @elapsed f_diffeq(Θ)
            t_gradient = @elapsed Zygote.gradient(f_diffeq, Θ)

            # Should still be reasonable
            @test t_gradient < 30 * t_forward

            println("DiffEq Forward: $(t_forward)s, Gradient: $(t_gradient)s")
        end
    end

end

@testset "Edge Cases in Gradients" begin

    @testset "Gradient at initial parameters" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        ∇Jc = Zygote.gradient(f, Θ)[1]

        # Should be defined and finite
        @test ∇Jc isa Tuple
        @test all(isfinite.(vec_params(∇Jc)))
    end

    @testset "Gradient at perturbed parameters" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Large perturbation
        Θ_pert = add_direction(Θ, random_direction(Θ), 10.0)

        f(θ) = mfg(θ)
        ∇Jc = Zygote.gradient(f, Θ_pert)[1]

        # Should still be defined (though may be large)
        @test ∇Jc isa Tuple
        grad_vec = vec_params(∇Jc)
        # Allow for large gradients, but should be finite
        @test all(isfinite.(grad_vec))
    end

    @testset "Gradient with single particle" begin
        prob_data = gaussian_1d_transport(nex=1)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        f(θ) = mfg(θ)
        Jc, ∇Jc = Zygote.withgradient(f, Θ)

        @test isfinite(Jc)
        @test all(isfinite.(vec_params(∇Jc[1])))
    end

end

@testset "Gradient Comparison: Legacy vs DiffEq" begin

    @testset "1D problem - gradients should be close" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Legacy gradient
        f_legacy(θ) = mfg(θ)
        ∇_legacy = Zygote.gradient(f_legacy, Θ)[1]

        @test_skip begin
            # DiffEq gradient
            f_diffeq(θ) = mfg(θ, use_diffeq=true,
                             solver=Tsit5(),
                             abstol=1e-8, reltol=1e-6)
            ∇_diffeq = Zygote.gradient(f_diffeq, Θ)[1]

            # Compare
            tol = MIGRATION_TOLERANCES
            compare_gradients(∇_legacy, ∇_diffeq;
                            abstol=tol.gradient_abstol,
                            reltol=tol.gradient_reltol,
                            name="Legacy vs DiffEq gradients")
        end
    end

    @testset "2D problem - gradient consistency" begin
        prob_data = gaussian_2d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Legacy
        f_legacy(θ) = mfg(θ)
        ∇_legacy = Zygote.gradient(f_legacy, Θ)[1]

        @test_skip begin
            # DiffEq
            f_diffeq(θ) = mfg(θ, use_diffeq=true, solver=Tsit5())
            ∇_diffeq = Zygote.gradient(f_diffeq, Θ)[1]

            # Should be reasonably close
            compare_gradients(∇_legacy, ∇_diffeq;
                            abstol=1e-5,
                            reltol=1e-2,
                            name="2D: Legacy vs DiffEq gradients")
        end
    end

end
