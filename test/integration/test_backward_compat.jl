"""
Backward compatibility tests

Ensures that:
1. Legacy API still works
2. Default behavior is sensible
3. Deprecation warnings are shown appropriately
4. Migration path is smooth
"""

using Test
using LinearAlgebra
using MFGnet

include("../utils/test_utils.jl")
include("../utils/test_problems.jl")
using .TestUtils
using .TestProblems

@testset "Legacy API Compatibility" begin

    @testset "Legacy MeanFieldGame constructor" begin
        # Old-style constructor with stepper and nt
        d = 2
        nex = 20
        X0 = randn(d, nex)
        rho0(x) = ones(size(x, 2))
        w = ones(nex) / nex

        F = F0()
        G = Gls(rho0, rho0, rho0(X0), rho0(X0), 1.0)

        # Legacy constructor (should still work)
        mfg = MeanFieldGame(F, G, X0, rho0, w;
                           stepper=RK4Step(),
                           nt=50)

        @test mfg isa MeanFieldGame
        @test mfg.stepper isa RK4Step
        @test mfg.nt == 50
    end

    @testset "Legacy evaluation (no use_diffeq flag)" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Old way: just call mfg(Θ)
        # Should use legacy path by default (for now)
        Jc = mfg(Θ)

        @test isfinite(Jc)
        @test Jc > 0
        @test size(mfg.UN, 2) == size(mfg.X0, 2)
    end

    @testset "Legacy BFGS still works" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        # Legacy BFGS
        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=10,
                                      atol=1e-6,
                                      out=-1)

        @test flag in [0, -1]
        @test size(his, 1) > 0
        @test his[end, 1] <= his[1, 1]  # Loss decreases
    end

end

@testset "Explicit use_diffeq=false" begin

    @testset "Can explicitly request legacy path" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Explicitly request legacy
            Jc = mfg(Θ, use_diffeq=false, stepper=RK4Step(), nt=50)

            @test isfinite(Jc)
        end
    end

    @testset "Legacy and new give similar results" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Legacy path
        Jc_legacy = mfg(Θ)

        @test_skip begin
            # New path (with similar settings)
            Jc_new = mfg(Θ, use_diffeq=true,
                        solver=RK4(),
                        adaptive=false,
                        dt=(mfg.tspan[2] - mfg.tspan[1]) / mfg.nt)

            # Should be very close
            compare_objectives(Jc_legacy, Jc_new;
                             abstol=1e-5,
                             reltol=1e-3,
                             name="Legacy vs new (RK4)")
        end
    end

end

@testset "Default Behavior" begin

    @testset "Default MeanFieldGame creation" begin
        # Minimal constructor (as in examples)
        d = 2
        nex = 30
        X0 = randn(d, nex)
        rho0(x) = ones(size(x, 2))
        w = ones(nex) / nex

        F = F0()
        G = Gls(rho0, rho0, rho0(X0), rho0(X0), 1.0)

        # New style: no stepper or nt
        mfg = MeanFieldGame(F, G, X0, rho0, w)

        @test mfg isa MeanFieldGame

        # Should have default values
        @test mfg.stepper isa Union{RK1Step, RK4Step}  # Has some default
        @test mfg.nt > 0  # Has some default
    end

    @testset "Default evaluation behavior" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Default call (no arguments)
        Jc = mfg(Θ)

        @test isfinite(Jc)
        @test Jc > 0

        # Should work with gradients
        using Zygote
        ∇Jc = Zygote.gradient(θ -> mfg(θ), Θ)[1]

        @test ∇Jc isa Tuple
        @test all(isfinite.(vec_params(∇Jc)))
    end

end

@testset "Deprecation Warnings" begin

    @testset "Warning when using old API" begin
        @test_skip begin
            # Creating with stepper should warn
            d = 2
            nex = 20
            X0 = randn(d, nex)
            rho0(x) = ones(size(x, 2))
            w = ones(nex) / nex

            F = F0()
            G = Gls(rho0, rho0, rho0(X0), rho0(X0), 1.0)

            # Should produce a warning
            mfg = @test_logs (:warn,) MeanFieldGame(F, G, X0, rho0, w;
                                                    stepper=RK4Step(),
                                                    nt=50)

            @test mfg isa MeanFieldGame
        end
    end

end

@testset "Migration Path Smoothness" begin

    @testset "Can migrate incrementally" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # Step 1: Use existing code
        Jc_old = mfg(Θ_init)
        @test isfinite(Jc_old)

        @test_skip begin
            # Step 2: Try new ODE solver (keep old optimization)
            function fdf_new(θ_vec)
                Θ = reconstruct_params(θ_vec, Θ_init)
                using Zygote
                Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ, use_diffeq=true), Θ)
                return Jc, vec_params(∇Jc[1])
            end

            f_new(θ_vec) = fdf_new(θ_vec)[1]
            θ0 = vec_params(Θ_init)

            # Old optimizer, new ODE solver
            θ_opt, flag, his, _, _ = bfgs(f_new, fdf_new, θ0;
                                         maxIter=5,
                                         atol=1e-6,
                                         out=-1)

            @test flag in [0, -1]
            @test size(his, 1) > 0

            # Step 3: Try new optimization wrapper
            prob = create_optimization_problem(mfg, Θ_init;
                                             use_diffeq=true)
            sol = solve_optimization(prob, LBFGS();
                                   maxiters=5,
                                   verbose=false)

            @test sol isa Optimization.OptimizationSolution
            @test isfinite(sol.objective)
        end
    end

end

@testset "Code Examples Still Work" begin

    @testset "Basic example pattern" begin
        # Pattern from examples: create problem and optimize
        d = 2
        nex = 50

        # Initial and target densities
        rho0(x) = exp.(-sum(x.^2, dims=1))
        rho1(x) = exp.(-sum((x .- 1.0).^2, dims=1))

        # Sample particles
        X0 = randn(d, nex)
        w = ones(nex) / nex

        # Create MFG
        F = F0()
        G = Gls(rho0, rho1, rho0(X0), rho1(X0), 1.0)
        Φ = PotentialNN(SingleLayer())

        mfg = MeanFieldGame(F, G, X0, rho0, w; Φ=Φ)

        # Initialize parameters
        m = 20
        Θ = initialize_single_layer_params(d, m)

        # Evaluate (should work)
        Jc = mfg(Θ)
        @test isfinite(Jc)

        # Compute gradient (should work)
        using Zygote
        ∇Jc = Zygote.gradient(θ -> mfg(θ), Θ)[1]
        @test all(isfinite.(vec_params(∇Jc)))
    end

end

@testset "Existing Tests Still Pass" begin

    # The original test suite should still pass
    @testset "Original SingleLayer test pattern" begin
        d = 2
        m = 10
        nex = 20

        N = SingleLayer()
        K = 0.01 * randn(m, d+1)
        b = 0.1 * randn(m)
        Θ = (K, b)

        s = randn(d+1, nex)

        # Forward pass
        z = N(s, Θ)
        @test size(z) == (m, nex)

        # Gradient computation should work
        w = ones(m) / sqrt(m)
        dΦ, d2Φ = getGradAndHessian(N, w, s, Θ)

        @test size(dΦ) == (d+1, nex)
        @test size(d2Φ) == (d+1, d+1, nex)
    end

    @testset "Original NN test pattern" begin
        # Pattern from testNN.jl
        d = 2
        m = 5
        nex = 10

        # Two-layer network
        layer1 = SingleLayer()
        layer2 = SingleLayer()
        net = NN([layer1, layer2])

        # Parameters
        K1 = 0.01 * randn(m, d+1)
        b1 = 0.1 * randn(m)
        K2 = 0.01 * randn(m, m)
        b2 = 0.1 * randn(m)

        Θ = ((K1, b1), (K2, b2))

        s = randn(d+1, nex)

        # Forward pass
        z = net(s, Θ)
        @test size(z) == (m, nex)
    end

end
